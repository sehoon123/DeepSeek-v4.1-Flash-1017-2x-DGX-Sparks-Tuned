#!/usr/bin/env python3
"""Build the optional sm_121 EP32 AOT set without CUDA initialization or a model.

The base set stays intact; eight checked cubins are appended. Run with Triton
3.7.1 from BASE_IMAGE and gemm_epilogue.py at /out/gemm_epilogue.py, as during
qualification. Refuse different compiled bytes rather than silently substituting
an unqualified compiler result. Native dispatch remains opt-in and row-bounded.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys

QUALIFIED = {
    (5120, 1280, '*bf16'): '32794a5845820993a711ee638b83960b3179b2656acc20c9cb79063ce687cdc5',
    (5120, 512, '*bf16'): 'b8d274dcbe6e18bd70eaa463b1520f2dc323b02c9ce1523d5cdab7cdeaeb0e18',
    (4096, 1024, '*bf16'): '15ab0e3b234fdceda5c8f593255afd2faeadd481dc203aa1c2b379ab01bb6c85',
    (1280, 16384, '*bf16'): 'ef861313ebeae7996e933b0ad9d67cca19ec38e7f91aa061bb67d94543ef37e1',
    (3072, 25600, '*fp32'): '675eef1c47e0b6526fb6ac884a29362a7b4bba882187683a1cc15c549b95f7be',
    (5120, 512, '*fp32'): '18a910016926ccaa1e4f8d69f8c8bd961de12979184b35c47d4dfced56cf85d2',
    (1280, 4096, '*bf16'): '672cfb0c593ab922a50f22f2e522badd0d28f135676f377a85a55a3527a2b485',
    (4096, 5120, '*fp32'): 'b5583b3f23e3d463c69dfaba110f7d2920b121e097343f07aeeaee8e8aa630d1',
}

def selected(base):
    kernels = base['kernels']
    if len(kernels) != 198 or len({e['hash'] for e in kernels}) != len(kernels):
        raise ValueError('expected the complete, unique pinned 198-entry kit')
    result = []
    found = set()
    for e in kernels:
        if not re.fullmatch('[0-9a-f]{12,64}', e['hash']):
            raise ValueError('invalid cubin filename hash')
        if e['fn'] == '_gemm_ep32r128':
            raise ValueError('base kit already contains the optional function')
        if e['fn'] != '_gemm':
            continue
        c = e['consts']
        shape = (c['K']['int'], c['N']['int'], e['params'][5]['type'])
        if shape not in QUALIFIED:
            continue
        if shape in found:
            raise ValueError('duplicate selected signature')
        found.add(shape)
        if any(c[k] != v for k, v in {
            'BM': {'int': 128}, 'BK': {'int': 32}, 'GROUP': {'int': 8},
            'HAS_BIAS': {'int': 0}, 'SCALE': {'f32': 1035273459},
        }.items()):
            raise ValueError('base specialization differs from qualification')
        if (e['num_warps'], e['num_ctas'], e['shared'], e['global_scratch'],
            e['profile_scratch'], e['pdl']) != (8, 1, 98304, 0, 0, False):
            raise ValueError('base launch/resource metadata differs')
        result.append((shape, e))
    if found != set(QUALIFIED):
        raise ValueError('missing selected GEMM signatures')
    return result

def build(base_dir, out):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    if triton.__version__ != '3.7.1':
        raise RuntimeError('this profile was qualified with Triton3.7.1')
    source = Path('/out/gemm_epilogue.py')
    # ptxas records the source file's mtime in both DWARF line tables. Preserve
    # the qualification timestamp for full cubin reproducibility, not just SASS.
    os.utime(source, (1793120689, 1793120689))
    spec = importlib.util.spec_from_file_location('gemm_epilogue', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = module._gemm_ep
    base = json.loads((base_dir / 'aot.json').read_text())
    choices = selected(base)
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(base_dir, out)
    original_count = len(base['kernels'])
    keys = list(base['kernels'][0])
    for shape, b in choices:
        consts = {k: (v['int'] if 'int' in v else
            struct.unpack('<f', struct.pack('<I', v['f32']))[0])
            for k, v in b['consts'].items()}
        consts.update(SCALE=0.08838834764831845, EP=32)
        signature = {p['name']: p['type'] for p in b['params']}
        signature.update({k: 'constexpr' for k in consts})
        attrs = {(fn.arg_names.index(p['name']),): [['tt.divisibility', 16]]
                 for p in b['params'] if p['div16']}
        kernel = triton.compile(ASTSource(fn, signature=signature,
            constexprs=consts, attrs=attrs), target=GPUTarget('cuda', 121, 32),
            options={'num_warps': 8, 'num_stages': 4, 'maxnreg': 128})
        md = kernel.metadata._asdict()
        data = kernel.asm['cubin']
        digest = hashlib.sha256(data).hexdigest()
        if digest != QUALIFIED[shape]:
            raise RuntimeError(f'unqualified compiled bytes for {shape}: {digest}')
        if (md['num_warps'], md.get('num_ctas', 1), md.get('shared', 0),
            md.get('global_scratch_size', 0), md.get('profile_scratch_size', 0),
            bool(md.get('launch_pdl', False)), md.get('maxnreg')) != (8, 1, 49152, 0, 0, False, 128):
            raise RuntimeError('compiled resources differ from qualification')
        entry = copy.deepcopy(b)
        entry.update(fn='_gemm_ep32r128', hash=kernel.hash, name=md['name'],
            num_warps=8, num_ctas=1, shared=49152, global_scratch=0,
            global_align=md.get('global_scratch_align', 1), profile_scratch=0, pdl=False)
        entry['consts']['EP'] = {'int': 32}
        if kernel.hash in {e['hash'] for e in base['kernels']}:
            raise RuntimeError('compiled hash collision')
        (out / 'cubins' / (kernel.hash + '.cubin')).write_bytes(data)
        base['kernels'].append({k: entry[k] for k in keys})
        print(json.dumps({'shape': shape, 'cubin_sha256': digest, 'hash': kernel.hash}), flush=True)
    if len(base['kernels']) != original_count + 8:
        raise RuntimeError('incomplete optional AOT set')
    base['generator'] = 'original deployed 198 entries + eight checked BM128/BK32/EP32/maxnreg128 variants'
    (out / 'aot.json').write_text(json.dumps(base, indent=1) + '\n')
    print('Complete: original198 + eight exact qualified cubins; no GPU/model initialization', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    build(args.base, args.out)
