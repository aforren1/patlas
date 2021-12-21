
import argparse


def main():
    from argparse import ArgumentParser
    from glob import glob
    from patlas import AtlasPacker
    
    parser = ArgumentParser(prog = 'patlas', description='Simple texture atlas packer.')

    parser.add_argument('files', nargs='+', help='List of files/paths with wildcards')
    parser.add_argument('-side', type=int, help='Length of one side of the square atlas')
    parser.add_argument('-pad', type=int, default=2, help='Padding between images')
    parser.add_argument('--visualize', action='store_true', help='Visualize the texture atlas')
    parser.add_argument('-o', type=str, help='Output filename/path')

    args = parser.parse_args()
    print(args.files)
    files = [glob(x) for x in args.files]
    files = [item for sublist in files for item in sublist]
    
    ap = AtlasPacker(side=args.side, pad=args.pad)
    ap.pack(files)

    if args.visualize:
        import matplotlib.pyplot as plt
        print(ap.metadata)
        plt.imshow(ap.atlas, origin='lower')
        plt.show()
    
    if args.o:
        ap.save(args.o)
