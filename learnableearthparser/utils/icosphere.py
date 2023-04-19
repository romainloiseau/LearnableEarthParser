from .utils_icosphere.icosphere import ico_sphere

def generate_icosphere(level: int = 0, device=None):
    icosphere = ico_sphere(level, device)
    return icosphere._verts_list[0], icosphere._faces_list[0]