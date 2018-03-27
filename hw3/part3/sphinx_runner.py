import argparse
import csv
import os
import re
import struct
import sys

import numpy as np

# Header for sphinx files
HEADER = """s3
version 0.1
mdef_file ../../../../sphinx/en_us.ci_cont/mdef
n_sen {n_states}
logbase {logbase:.06f}
endhdr
"""

# Command to run sphinx
SPHINX = """pocketsphinx_batch \
        -hmm {hmm} \
        -lm {lm} \
        -cepdir {cepdir} \
        -cepext {ext} \
        -hyp {hyp} \
        -ctl {ctl} \
        -dict {dict} \
        -compallsen yes \
        -pl_window 0 \
        -senin yes \
        -logbase {logbase}
"""

# File names
CTL = 'sen.txt'
HYP = 'hyp.txt'
SEN = 'sen-{:08d}'
EXT = '.sen'

# Number of states
N_STATES = 46 * 3


def write_senones(args):
    # Read logits and write sphinx data files
    if not os.path.exists(args.logits):
        print("Error: missing file {}".format(args.logits))
        exit(-1)
    logits = np.load(args.logits)
    write_logits(logits, args)


def write_logits(logits, args):
    # Write sphinx data files and control file
    os.makedirs(args.senones, exist_ok=True)
    with open(os.path.join(args.senones, CTL), 'w') as f:
        for i, logit in enumerate(logits):
            file = SEN.format(i)
            f.write(file + "\n")
            write_logit(os.path.join(args.senones, file + EXT), logit, args)


def convert_logit(logit, args):
    # Convert logits from network to sphinx scaling
    logit = logit * -1 * args.logit_scale
    logit = logit - np.min(logit, axis=1, keepdims=True)
    return logit.astype(np.int32)


def write_logit(filename, logit, args):
    # Write single sphinx data file
    n_active = logit.shape[1]
    assert n_active == N_STATES
    with open(filename, 'wb') as f:
        f.write(HEADER.format(logbase=args.logbase, n_states=N_STATES).encode('ascii'))
        f.write(struct.pack('I', 0x11223344))
        for r in convert_logit(logit, args):
            f.write(struct.pack('h', n_active))
            f.write(struct.pack('%sh' % len(r), *r))


def run_sphinx(args):
    # Run sphinx command line
    cmd = SPHINX.format(
        hmm=args.hmm,
        lm=args.lm,
        cepdir=args.senones,
        hyp=os.path.join(args.senones, HYP),
        ctl=os.path.join(args.senones, CTL),
        dict=args.dict,
        logbase=args.logbase,
        ext=EXT
    )
    print(cmd)
    os.system(cmd)


def format_submission(args):
    # Format the submission for Kaggle
    os.makedirs(os.path.dirname(os.path.abspath(args.submission)), exist_ok=True)
    with open(args.submission, 'w', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(['Id', 'Predicted'])
        with open(os.path.join(args.senones, HYP)) as fin:
            for i, line in enumerate(fin):
                line = re.sub(r'\(.*\)', '', line).strip()
                if len(line) > 0:
                    w.writerow([i, line])


def run(args):
    # Read logits file and write into sphinx data files
    if not args.no_write_senones:
        write_senones(args)
    # Run sphinx on data files
    if not args.no_run_sphinx:
        run_sphinx(args)
    # Parse results for Kaggle
    if not args.no_format_submission:
        format_submission(args)


def main(argv):
    parser = argparse.ArgumentParser(description='Sphinx Runner')

    # Input and output paths
    parser.add_argument('--logits', type=str, default='logits.npy', help='Input Logits')
    parser.add_argument('--senones', type=str, default='senones', help='Senone Path')
    parser.add_argument('--hypothesis', type=str, default='hypothesis.txt', help='Senone Path')
    parser.add_argument('--submission', type=str, default='submission.csv', help='Senone Path')

    # Sphinx configuration
    parser.add_argument('--hmm', type=str, default='asr_files/en_us.ci_cont', help='HMM path')
    parser.add_argument('--lm', type=str, default='asr_files/tcb20onp.Z.DMP', help='LM path')
    parser.add_argument('--dict', type=str, default='asr_files/cmudict.0.6d.wsj0', help='Dict path')
    parser.add_argument('--logbase', type=float, default=1.0001, help='logbase')
    parser.add_argument('--logit-scale', type=float, default=10, help='scale for logits')

    # Enable/Disable steps
    parser.add_argument('--no-write-senones', default=False, action='store_true',
                        help='do not write senones')
    parser.add_argument('--no-run-sphinx', default=False, action='store_true',
                        help='do not run sphinx')
    parser.add_argument('--no-format-submission', default=False, action='store_true',
                        help='do not format submission')

    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
