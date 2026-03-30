#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --account=project_2005815
#SBATCH --output=logs/train_multi_%j.out
#SBATCH --error=logs/train_multi_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100:4

set -x
set -euo pipefail

module load gcc cuda

# # Prepare train data

# # Low-res
# # Es - ara
# cp data/lowres/train.spa-ara.ara "data/multi/train.spa-ara.ara"
# sed "s/^/>>spa<< >>ara<< /" data/lowres/train.spa-ara.spa > "data/multi/train.spa-ara.spa"

# cp data/lowres/train.spa-ara.spa "data/multi/train.ara-spa.spa"
# sed "s/^/>>ara<< >>spa<< /" "data/multi/train.spa-ara.ara" > "data/multi/train.ara-spa.ara"


# # Es - arg
# cp data/lowres/train.spa-arg.arg "data/multi/train.spa-arg.arg"
# sed "s/^/>>spa<< >>arg<< /" data/lowres/train.spa-arg.spa > "data/multi/train.spa-arg.spa"

# cp data/lowres/train.spa-arg.spa "data/multi/train.arg-spa.spa"
# sed "s/^/>>arg<< >>spa<< /" "data/multi/train.spa-arg.arg" > "data/multi/train.arg-spa.arg"

# # Es - ast
# cp data/lowres/train.spa-ast.ast "data/multi/train.spa-ast.ast"
# sed "s/^/>>spa<< >>ast<< /" data/lowres/train.spa-ast.spa > "data/multi/train.spa-ast.spa"

# cp data/lowres/train.spa-ast.spa "data/multi/train.ast-spa.spa"
# sed "s/^/>>ast<< >>spa<< /" "data/multi/train.spa-ast.ast" > "data/multi/train.ast-spa.ast"

# # Es - oci
# cp data/lowres/train.spa-oci.oci "data/multi/train.spa-oci.oci"
# sed "s/^/>>spa<< >>oci<< /" data/lowres/train.spa-oci.spa > "data/multi/train.spa-oci.spa"

# cp data/lowres/train.spa-oci.spa "data/multi/train.oci-spa.spa"
# sed "s/^/>>oci<< >>spa<< /" "data/multi/train.spa-oci.oci" > "data/multi/train.oci-spa.oci"

# # Es-eo
# cp data/es-eo/clean/train.eo "data/multi/train.spa-epo.epo"
# sed "s/^/>>spa<< >>epo<< /" data/es-eo/clean/train.es > "data/multi/train.spa-epo.spa"

# cp data/es-eo/clean/train.es "data/multi/train.epo-spa.spa"
# sed "s/^/>>epo<< >>spa<< /" data/es-eo/clean/train.eo > "data/multi/train.epo-spa.epo"

# # ca-eo

# cp data/ca-eo/clean/train.eo "data/multi/train.cat-epo.epo"
# sed "s/^/>>cat<< >>epo<< /" data/ca-eo/clean/train.ca > "data/multi/train.cat-epo.cat"

# cp data/ca-eo/clean/train.ca "data/multi/train.epo-cat.cat"
# sed "s/^/>>epo<< >>cat<< /" data/ca-eo/clean/train.eo > "data/multi/train.epo-cat.epo"

# # Concatenate

TRAIN="data/multi/train.multi.tsv"
# SAMPLES=300000
# # Start fresh
# rm -f "$TRAIN"

# # spa -> ara
# paste data/multi/train.spa-ara.spa  data/multi/train.spa-ara.ara | shuf -n $SAMPLES >> "$TRAIN"

# # ara -> spa
# paste data/multi/train.ara-spa.ara  data/multi/train.ara-spa.spa | shuf -n $SAMPLES >> "$TRAIN"

# # spa -> arg
# paste data/multi/train.spa-arg.spa  data/multi/train.spa-arg.arg | shuf -n $SAMPLES >> "$TRAIN"

# # arg -> spa
# paste data/multi/train.arg-spa.arg  data/multi/train.arg-spa.spa | shuf -n $SAMPLES >> "$TRAIN"

# # spa -> ast
# paste data/multi/train.spa-ast.spa  data/multi/train.spa-ast.ast | shuf -n $SAMPLES >> "$TRAIN"

# # ast -> spa
# paste data/multi/train.ast-spa.ast  data/multi/train.ast-spa.spa | shuf -n $SAMPLES >> "$TRAIN"

# # spa -> oci
# paste data/multi/train.spa-oci.spa  data/multi/train.spa-oci.oci | shuf -n $SAMPLES >> "$TRAIN"

# # oci -> spa
# paste data/multi/train.oci-spa.oci  data/multi/train.oci-spa.spa | shuf -n $SAMPLES >> "$TRAIN"

# # spa -> epo
# paste data/multi/train.spa-epo.spa  data/multi/train.spa-epo.epo | shuf -n $SAMPLES >> "$TRAIN"

# # epo -> spa
# paste data/multi/train.epo-spa.epo  data/multi/train.epo-spa.spa | shuf -n $SAMPLES >> "$TRAIN"

# # cat -> epo
# paste data/multi/train.cat-epo.cat  data/multi/train.cat-epo.epo | shuf -n $SAMPLES >> "$TRAIN"

# # epo -> cat
# paste data/multi/train.epo-cat.epo  data/multi/train.epo-cat.cat | shuf -n $SAMPLES >> "$TRAIN"

# # Prepare dev data
# # Es - ara
# # Prepare dev data

# # Es - ara
# cp data/dev.ara "data/multi/dev.spa-ara.ara"
# sed "s/^/>>spa<< >>ara<< /" data/dev.spa > "data/multi/dev.spa-ara.spa"

# cp data/dev.spa "data/multi/dev.ara-spa.spa"
# sed "s/^/>>ara<< >>spa<< /" data/dev.ara > "data/multi/dev.ara-spa.ara"

# # Es - arg
# cp data/dev.arg "data/multi/dev.spa-arg.arg"
# sed "s/^/>>spa<< >>arg<< /" data/dev.spa > "data/multi/dev.spa-arg.spa"

# cp data/dev.spa "data/multi/dev.arg-spa.spa"
# sed "s/^/>>arg<< >>spa<< /" data/dev.arg > "data/multi/dev.arg-spa.arg"

# # Es - ast
# cp data/dev.ast "data/multi/dev.spa-ast.ast"
# sed "s/^/>>spa<< >>ast<< /" data/dev.spa > "data/multi/dev.spa-ast.spa"

# cp data/dev.spa "data/multi/dev.ast-spa.spa"
# sed "s/^/>>ast<< >>spa<< /" data/dev.ast > "data/multi/dev.ast-spa.ast"

# # Es - oci
# cp data/dev.oci "data/multi/dev.spa-oci.oci"
# sed "s/^/>>spa<< >>oci<< /" data/dev.spa > "data/multi/dev.spa-oci.spa"

# cp data/dev.spa "data/multi/dev.oci-spa.spa"
# sed "s/^/>>oci<< >>spa<< /" data/dev.oci > "data/multi/dev.oci-spa.oci"

# # Es-eo
# cp data/dev.epo "data/multi/dev.spa-epo.epo"
# sed "s/^/>>spa<< >>epo<< /" data/dev.spa > "data/multi/dev.spa-epo.spa"

# cp data/dev.spa "data/multi/dev.epo-spa.spa"
# sed "s/^/>>epo<< >>spa<< /" data/dev.epo > "data/multi/dev.epo-spa.epo"

# # ca-eo
# cp data/dev.epo "data/multi/dev.cat-epo.epo"
# sed "s/^/>>cat<< >>epo<< /" data/dev.cat > "data/multi/dev.cat-epo.cat"

# cp data/dev.cat "data/multi/dev.epo-cat.cat"
# sed "s/^/>>epo<< >>cat<< /" data/dev.epo > "data/multi/dev.epo-cat.epo"

# # Concatenate
DEV="data/multi/dev.multi.tsv"
# SAMPLES_DEV=333

# # Start fresh
# rm -f "$DEV"

# # spa -> ara
# paste data/multi/dev.spa-ara.spa  data/multi/dev.spa-ara.ara | shuf -n $SAMPLES_DEV >> "$DEV"

# # ara -> spa
# paste data/multi/dev.ara-spa.ara  data/multi/dev.ara-spa.spa | shuf -n $SAMPLES_DEV >> "$DEV"

# # spa -> arg
# paste data/multi/dev.spa-arg.spa  data/multi/dev.spa-arg.arg | shuf -n $SAMPLES_DEV >> "$DEV"

# # ara -> spa
# paste data/multi/dev.arg-spa.arg  data/multi/dev.arg-spa.spa | shuf -n $SAMPLES_DEV >> "$DEV"

# # spa -> ast
# paste data/multi/dev.spa-ast.spa  data/multi/dev.spa-ast.ast | shuf -n $SAMPLES_DEV >> "$DEV"

# # ast -> spa
# paste data/multi/dev.ast-spa.ast  data/multi/dev.ast-spa.spa | shuf -n $SAMPLES_DEV >> "$DEV"

# # spa -> oci
# paste data/multi/dev.spa-oci.spa  data/multi/dev.spa-oci.oci | shuf -n $SAMPLES_DEV >> "$DEV"

# # oci -> spa
# paste data/multi/dev.oci-spa.oci  data/multi/dev.oci-spa.spa | shuf -n $SAMPLES_DEV >> "$DEV"

# # spa -> epo
# paste data/multi/dev.spa-epo.spa  data/multi/dev.spa-epo.epo | shuf -n $SAMPLES_DEV >> "$DEV"

# # epo -> spa
# paste data/multi/dev.epo-spa.epo  data/multi/dev.epo-spa.spa | shuf -n $SAMPLES_DEV >> "$DEV"

# # cat -> epo
# paste data/multi/dev.cat-epo.cat  data/multi/dev.cat-epo.epo | shuf -n $SAMPLES_DEV >> "$DEV"

# # epo -> cat
# paste data/multi/dev.epo-cat.epo  data/multi/dev.epo-cat.cat | shuf -n $SAMPLES_DEV >> "$DEV"

#-------- Run Marian training --------

vocab="models/multi/vocab.spm"

/scratch/project_2005815/members/degibert/MTM25/marian/build/marian -c scripts/marian_config.yml \
  --model "models/multi/model.npz" \
  --train-sets "$TRAIN" \
  --valid-sets "$DEV" \
  --vocabs "$vocab" "$vocab" \
  --log "models/multi/train.log" \
  --valid-log "models/multi/valid.log" \
  --devices 0 1 2 3 --tsv