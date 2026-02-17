"""Evaluation utilities.

The primary evaluation is done via CBF-QP safety filtering using:
  - A BC reference policy
  - The learned Vc as a Control Barrier Function
  - A learned control-affine dynamics model

See jaxrl5.agents.vocbf.vocbf for evaluate_policy() and evaluate_cbf().
"""
