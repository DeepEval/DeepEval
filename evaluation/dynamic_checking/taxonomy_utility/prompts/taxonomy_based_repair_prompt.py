class TaxonomyBasedRepairPrompt:
    def __init__(self):
        pass

    def prompt(self, buggy_code, messages):
        background = """
You are a Python code runtime error repair expert. Your task is to help fix the provided Python code based on the compile log.
"""

        instruction = """
Check the above origin code and compile log, fix the errors according to the instruction below, and output the corrected code.

Follow this taxonomy-guided fix procedure for Tensor Shape Mismatch Error:
(1) From the compile log, locate the first failing merge operator (Add/Concatenate/Multiply) and the tensors feeding into it.
(2) Trace and extract the two (or more) tensor flow paths that produce those tensors.
(3) Perform shape-aware analysis along each extracted path: 
   - Treat shapes explicitly shown in the compile log as ground truth anchors; 
   - Infer intermediate shapes only when necessary to determine where the fix should be applied.
(4) Determine the mismatch type:
   - Spatial mismatch: inconsistent Height/Width;
   - Channel mismatch: inconsistent Channel;
(5) Repair with minimal edits near the merge:
   (i) For spatial mismatch:
   - Strides (stride_h, stride_w):
     - Larger stride -> smaller H_out / W_out (approximately scaled by 1 / stride).
     - Smaller stride -> larger H_out / W_out.
   - Padding:
     - padding="same" preserves spatial size when stride = 1.
     - padding="valid" reduces spatial size.
   - Window size (kernel_size for Conv layers; pool_size for Pooling layers):
     - Larger window -> smaller H_out / W_out under padding="valid".
     - Smaller window -> larger H_out / W_out under padding="valid".
     
   Modify only parameters of existing layers on the shortest path to the failing merge. Allowed edits:
     - Conv layers: modify {strides, padding}
     - Pooling layers: modify {pool_size, strides, padding}

   (ii) For channel mismatch: Allowed edits:
     - Conv layers: modify {filters}
     - Dense layers: modify {units}

Note: Do not introduce any new layers; no additional layers.X(...) or tf.keras.layers.X(...) statements beyond those already in the original code.
"""
        prompt = background + f"\n\nThe Origin Code:\n```python\n{buggy_code}```\n\nCompile Log:\n{messages}" + f"\n{instruction}\n "  

        return prompt