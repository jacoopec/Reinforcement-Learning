It’s highlighting a model-based viewpoint:

Instead of directly averaging returns (pure Monte Carlo),

you first build an estimated model (transition/reward table),

then compute values from that model (like dynamic programming on the learned model).

If you want, I can show how you’d compute these values using the Bellman equations step-by-step, or how MC and TD would estimate them directly without building the model.
