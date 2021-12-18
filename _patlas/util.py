
import argparse


def main():
    from argparse import ArgumentParser
    from glob import glob
    from patlas import AtlasPacker
    
    parser = ArgumentParser(prog = 'patlas', description='Simple texture atlas packer.')

    parser.add_argument('files', nargs='+', help='List of files/wildcard things')
    parser.add_argument('-side', type=int)
    parser.add_argument('-pad', type=int, default=2)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    print(args.files)
    files = [glob(x) for x in args.files]
    files = [item for sublist in files for item in sublist]
    
    ap = AtlasPacker(side=args.side, pad=args.pad)
    ap.pack(files)

    if args.visualize:
        import matplotlib.pyplot as plt
        print(ap.locations)
        plt.imshow(ap.atlas, origin='lower')
        plt.show()
