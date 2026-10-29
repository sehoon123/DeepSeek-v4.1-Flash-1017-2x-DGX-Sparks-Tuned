#!/usr/bin/env python3
"""DISPLAY_EMBED's probe (zig/patches/0056), run by start.sh on each Spark in the engine's image, as root with the card.

Can this Spark's display reservation hold a table of BYTES? nvidia-drm with modeset on, no display connected to CARD,
a DRM dumb buffer of BYTES (in whole 16 MiB pieces) and its dma-buf export, then both released at once. No CUDA, no GPU
work: the buffer comes from the firmware's scanout carve-out, not from Linux's memory. Prints one line; exit 0: usable.

usage: display_probe.py BYTES [CARD]   (CARD: /dev/dri/card0)
"""
import fcntl, glob, os, struct, sys

CREATE_DUMB = 0xC02064B2   # DRM_IOWR(0xB2, struct drm_mode_create_dumb)
DESTROY_DUMB = 0xC00464B4  # DRM_IOWR(0xB4, struct drm_mode_destroy_dumb)
PRIME_TO_FD = 0xC00C642D   # DRM_IOWR(0x2D, struct drm_prime_handle)
WIDTH, BPP, GRANULE = 4096, 32, 16 << 20


def main():
    want = int(sys.argv[1])
    card = sys.argv[2] if len(sys.argv) > 2 else '/dev/dri/card0'
    want = -(-want // GRANULE) * GRANULE
    try:
        modeset = open('/sys/module/nvidia_drm/parameters/modeset').read().strip()
    except OSError as e:
        return f'no nvidia_drm modeset parameter ({e})'
    if modeset != 'Y':
        return 'nvidia-drm runs without modeset (dumb buffers need it: options nvidia-drm modeset=1)'
    name = os.path.basename(card)
    for p in glob.glob(f'/sys/class/drm/{name}-*/status'):
        if open(p).read().strip() == 'connected':
            return f'a display is connected ({os.path.basename(os.path.dirname(p))}): the reservation is its screen\'s'
    try:
        fd = os.open(card, os.O_RDWR | os.O_CLOEXEC)
    except OSError as e:
        return f'{card}: {e}'
    try:
        req = bytearray(struct.pack('IIIIIIQ', want // (WIDTH * BPP // 8), WIDTH, BPP, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, CREATE_DUMB, req)
        except OSError as e:
            return f'a {want >> 20} MiB dumb buffer on {card}: {e} (the reservation has less room, or holds another span)'
        handle, size = struct.unpack('IIIIIIQ', req)[4], struct.unpack('IIIIIIQ', req)[6]
        try:
            if size < want:
                return f'a dumb buffer of {size} bytes, asked for {want}'
            prime = bytearray(struct.pack('IIi', handle, os.O_CLOEXEC | os.O_RDWR, -1))
            try:
                fcntl.ioctl(fd, PRIME_TO_FD, prime)
            except OSError as e:
                return f'the dumb buffer\'s dma-buf export: {e}'
            os.close(struct.unpack('IIi', prime)[2])
        finally:
            fcntl.ioctl(fd, DESTROY_DUMB, bytearray(struct.pack('I', handle)))
    finally:
        os.close(fd)
    print(f'ok: {card} can hold {want >> 20} MiB of the display reservation')
    return None


if __name__ == '__main__':
    why = main()
    if why:
        print(why)
        sys.exit(1)
