# Wrapper around onnxruntime_genai.models.builder for a text-only Qwen3_5ForCausalLM checkpoint.
# The 0.18.0 builder dispatches Qwen3.5 only on architectures[0]=="Qwen3_5ForConditionalGeneration";
# we relabel the architecture and make its weight loader use Qwen3_5ForCausalLM (same text weights, no vision tower).
import sys, os, time
import onnxruntime_genai.models as M
sys.path.insert(0, os.path.dirname(M.__file__))
from onnxruntime_genai.models import builder
import builders.base as bb
from transformers import Qwen3_5ForCausalLM
bb.Qwen3_5ForConditionalGeneration = Qwen3_5ForCausalLM
sys.argv = [sys.argv[0]] + sys.argv[1:]
args = builder.get_args()
extra = builder.parse_extra_options(args.model_name, args.input, args.output, args.precision,
                                    args.execution_provider, args.cache_dir, args.extra_options)
cfg = extra["hf_details"]["hf_config"]
print("orig architectures:", cfg.architectures, "model_type:", cfg.model_type)
cfg.architectures = ["Qwen3_5ForConditionalGeneration"]
t = time.time()
builder.create_model(args.model_name, args.input, args.output, args.precision, args.execution_provider, args.cache_dir, **extra)
print("BUILD_SECONDS", round(time.time() - t, 1))
