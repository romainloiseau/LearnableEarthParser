import math
import numpy as np

def vplane(N):
    xy = np.random.uniform(-1, 1, (N, 2))
    z = np.zeros((N, 1))
    plane = np.concatenate((xy, z), axis=-1)
    return plane.astype(np.float32)

def hplane(N):
    yz = np.random.uniform(-1, 1, (N, 2))
    x = np.zeros((N, 1))
    plane = np.concatenate((x, yz), axis=-1)
    return plane.astype(np.float32)

def vline(N):
    xy = np.zeros((N, 2))
    z = np.random.uniform(-1, 1, (N, 1))
    line = np.concatenate((xy, z), axis=-1)
    return line.astype(np.float32)

def hline(N):
    yz = np.zeros((N, 2))
    x = np.random.uniform(-1, 1, (N, 1))
    line = np.concatenate((x, yz), axis=-1)
    return line.astype(np.float32)

def ball(N):
    phi = np.random.uniform(0, 2*math.pi, (N, 1))
    r = np.random.uniform(0, 1, (N, 1))**(1/3)
    cos_theta = np.random.uniform(-1, 1, (N, 1))
    x = r * np.sqrt(1-cos_theta**2) * np.cos(phi)
    y = r * np.sqrt(1-cos_theta**2) * np.sin(phi)
    z = r * cos_theta

    return np.concatenate([x, y, z], -1).astype(np.float32)

def sphere(N):
    xyz = np.random.normal(0, 1, (3, N))
    xyz /= (xyz**2).sum(0)**.5
    return xyz.T.astype(np.float32)

def half_sphere(N):
    xyz = sphere(N)
    xyz[:, -1] *= np.sign(xyz[:, -1])
    return xyz.astype(np.float32)

def roof(N):
    xyz = np.random.uniform(0, 2, (N, 3))
    side = np.random.randint(0, 2, N).astype(np.bool_)
    xyz[side, 1] *= 0
    xyz[~side, 2] *= 0

    cp4, sp4 = math.cos(math.pi / 4), math.sin(math.pi / 4)
    xyz = xyz.dot(np.array([[1, 0, 0], [0, -cp4, -sp4], [0, sp4, -cp4]]))
    xyz -= xyz.mean(0)
    return xyz.astype(np.float32)

def plane(N):
    xy = np.random.uniform(-1, 1, (N, 2))
    z = np.zeros((N, 1))
    plane = np.concatenate((xy, z), axis=-1)
    return plane.astype(np.float32)

def box(N):
    box = np.random.uniform(-1, 1, (N, 3))
    face = np.random.randint(0, 6, N)
    for i in range(6):
        box[face == i, i // 2] = 2 * (i%2) - 1
    box /= ((box**2).sum(-1)**.5).max()
    return box.astype(np.float32)

def cube(N):
    cube = np.random.uniform(-1, 1, (N, 3))
    return cube.astype(np.float32)

def cylinder(N):
    n = int(N / 2)
    #disks
    r = np.random.uniform(0, 1, (n, 1))**.5
    theta = 2*math.pi * np.random.uniform(0, 1, (n, 1))
    disks = np.hstack([
        r * np.cos(theta),
        r * np.sin(theta),
        .5*np.ones((n, 1))
    ])
    disks[np.random.randint(0, 2, n).astype(bool), -1] *= -1
    # surface
    theta = 2*math.pi * np.random.uniform(0, 1, (n, 1))
    surface = np.hstack([
        np.cos(theta),
        np.sin(theta),
        np.random.uniform(-0.5, 0.5, (n, 1))
    ])
    cylinder = np.concatenate([
        disks,
        surface
    ], 0)
    
    cylinder -= cylinder.mean(0)
    cylinder /= ((cylinder**2).sum(-1)**.5).max()
    
    return cylinder.astype(np.float32)

def tetrahedron(N):
    triangle = np.hstack([
        np.random.triangular(-1, 0, 1, (N, 1)),
        np.zeros((N, 2))
    ])
    cp6, sp6 = math.cos(math.pi / 6), math.sin(math.pi / 6)
    triangle[:, 1] = (1-np.abs(triangle[:, 0])) * np.random.uniform(0, 2*cp6, N)
    face = np.random.randint(0, 4, N)
    facei = face != 0
    triangle[facei, 2] = triangle[facei, 1]
    triangle[facei, 1] /= 3.
    for i in range(2, 4):
        facei = face >= i
        triangle[facei] = triangle[facei].dot(np.array([[-sp6, cp6, 0], [-cp6, -sp6, 0], [0, 0, 1]]))
        triangle[facei, 1] += cp6
        triangle[facei, 0] += sp6
    triangle -= triangle.mean(0)
    triangle /= ((triangle**2).sum(-1)**.5).max()
    
    return triangle.astype(np.float32)