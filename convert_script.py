import sys

in_file = sys.argv[1]
out_file = in_file.split(".sh")[0] + "_mod.sh"

with open(in_file, 'r') as in_data, open(out_file, 'w') as out_data:
    for line in in_data:
        if line.startswith("python"):
            args = line.split("&&")[0].split()
            if len(args) == 10:
                py, command, model, device, out_dir, random, attractor, seed, examples, language = args
                intervention = "natural"
            elif len(args) == 11:
                py, command, model, device, out_dir, random, attractor, seed, examples, language, intervention = args

            new_command = f"{py} {command} {model} {attractor} {intervention} language:{language} && wait"
            out_data.write(new_command + "\n")

        else:
            out_data.write(line)