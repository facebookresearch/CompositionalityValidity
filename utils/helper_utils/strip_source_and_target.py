# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strip targets from a tsv file and write as newline-separated txt.

This file can be useful as input to generate predictions (e.g. for evaluation).
"""
import sys
import os

from absl import app
from absl import flags
base_dir = os.getenv("BASE_DIR")

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tsv file.")

flags.DEFINE_string("output_source", "", "Output source txt file.")
flags.DEFINE_string("output_target", "", "Output target txt file.")

flags.DEFINE_string("prefix", "", "Optional prefix to prepend to source.")

# python utils/helper_utils/strip_source_and_target.py --input="data/COGS/no_mod_split/nqg_train.tsv" --output_source="data/COGS/no_mod_split/btg_source.txt" --output_target="data/COGS/no_mod_split/btg_target.txt"
def read_tsv(filename, expected_num_columns=2):
  """Read file to list of examples."""
  examples = []
  with gfile.GFile(filename, "r") as tsv_file:
    for line in tsv_file:
      line = line.rstrip()
      cols = line.split("\t")
      examples.append(cols)
  print("Loaded %s examples from %s." % (len(examples), filename))
  return examples

def main(unused_argv):
  examples = read_tsv(FLAGS.input)
  with gfile.GFile(FLAGS.output_source, "w") as txt_source:
    with gfile.GFile(FLAGS.output_target, "w") as txt_target:
        for example in examples:
            txt_source.write("%s%s\n" % (FLAGS.prefix, example[0]))
            txt_target.write("%s%s\n" % (FLAGS.prefix, example[1]))


if __name__ == "__main__":
  app.run(main)
