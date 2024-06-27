This is a simple variant of the blocks world where each block is either
positive or negative and there are two hands, one positive and one
negative. The twist is that if a block is picked up by a hand of
opposite polarity then it is damaged such that no other block can be
placed on it, which can lead to deadends. The interaction between hands
and blocks of the same polarity is just as in the standard blocks
world.

The matching-bw generator is a shell script that calls three
executables.

Compile the executables with "make".

You also need to make sure that the shell script is executable via:

  chmod +x matching-bw-generator.sh

Then to generate a matching-bw problem with base name "bname" and size
"n" you can call:

  ./matching-bw-generator.sh bname n

This will create two files: bname-typed.pddl and bname-untyped.pddl

which are the typed and untyped versions of the files.
