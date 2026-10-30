from . import clip_frames, face_pass, frame_rate, fun_control_chain, object_chain, object_swap, scail_chain, streaming_finish, wan_chain

_modules = (wan_chain, fun_control_chain, scail_chain, frame_rate, face_pass, streaming_finish, object_swap, clip_frames, object_chain)
NODE_CLASS_MAPPINGS = {k: v for m in _modules for k, v in m.NODE_CLASS_MAPPINGS.items()}
NODE_DISPLAY_NAME_MAPPINGS = {k: v for m in _modules for k, v in m.NODE_DISPLAY_NAME_MAPPINGS.items()}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
