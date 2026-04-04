#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 08:05:01 2026

@author: caiazzo
"""

#!/usr/bin/env python3
"""Convert a Medit .mesh tetrahedral volume mesh into a hex-only volume mesh.

The conversion uses a conforming subdivision based on:
- edge midpoints
- face centroids
- tetrahedron centroids

Each tetrahedron is converted into 4 hexahedra.
Boundary triangles are converted to boundary quadrilaterals (3 per triangle).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MeshData:
    vertices: list[tuple[float, float, float, int]]
    triangles: list[tuple[int, int, int, int]]
    tetrahedra: list[tuple[int, int, int, int, int]]


def _next_nonempty(lines: list[str], i: int) -> int:
    n = len(lines)
    while i < n and not lines[i].strip():
        i += 1
    return i


def read_medit_mesh(path: Path) -> MeshData:
    lines = path.read_text().splitlines()

    vertices: list[tuple[float, float, float, int]] = []
    triangles: list[tuple[int, int, int, int]] = []
    tetrahedra: list[tuple[int, int, int, int, int]] = []

    i = 0
    n = len(lines)
    while i < n:
        i = _next_nonempty(lines, i)
        if i >= n:
            break

        token = lines[i].strip()
        i += 1

        if token in {"MeshVersionFormatted", "Dimension"}:
            i = _next_nonempty(lines, i)
            if i < n:
                i += 1
            continue

        if token == "Vertices":
            i = _next_nonempty(lines, i)
            nv = int(lines[i].strip())
            i += 1
            for _ in range(nv):
                i = _next_nonempty(lines, i)
                x, y, z, r = lines[i].split()[:4]
                vertices.append((float(x), float(y), float(z), int(r)))
                i += 1
            continue

        if token == "Triangles":
            i = _next_nonempty(lines, i)
            nt = int(lines[i].strip())
            i += 1
            for _ in range(nt):
                i = _next_nonempty(lines, i)
                a, b, c, r = lines[i].split()[:4]
                triangles.append((int(a), int(b), int(c), int(r)))
                i += 1
            continue

        if token == "Tetrahedra":
            i = _next_nonempty(lines, i)
            nt = int(lines[i].strip())
            i += 1
            for _ in range(nt):
                i = _next_nonempty(lines, i)
                a, b, c, d, r = lines[i].split()[:5]
                tetrahedra.append((int(a), int(b), int(c), int(d), int(r)))
                i += 1
            continue

        if token == "End":
            break

        # Unknown section: read count line then skip entries conservatively.
        i = _next_nonempty(lines, i)
        if i >= n:
            break
        try:
            count = int(lines[i].strip())
        except ValueError:
            continue
        i += 1
        for _ in range(count):
            i = _next_nonempty(lines, i)
            if i < n:
                i += 1

    if not vertices or not tetrahedra:
        raise ValueError(f"Input mesh '{path}' must contain Vertices and Tetrahedra sections")

    return MeshData(vertices=vertices, triangles=triangles, tetrahedra=tetrahedra)


def convert_tet_to_hex(mesh: MeshData):
    coords: list[tuple[float, float, float]] = [(x, y, z) for x, y, z, _ in mesh.vertices]
    refs: list[int] = [r for _, _, _, r in mesh.vertices]

    edge_nodes: dict[tuple[int, int], int] = {}
    face_nodes: dict[tuple[int, int, int], int] = {}

    def add_vertex(x: float, y: float, z: float, ref: int = 0) -> int:
        coords.append((x, y, z))
        refs.append(ref)
        return len(coords)  # 1-based id

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in edge_nodes:
            return edge_nodes[key]
        x1, y1, z1 = coords[i - 1]
        x2, y2, z2 = coords[j - 1]
        nid = add_vertex((x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5)
        edge_nodes[key] = nid
        return nid

    def face_center(i: int, j: int, k: int) -> int:
        key = tuple(sorted((i, j, k)))
        if key in face_nodes:
            return face_nodes[key]
        x1, y1, z1 = coords[i - 1]
        x2, y2, z2 = coords[j - 1]
        x3, y3, z3 = coords[k - 1]
        nid = add_vertex(
            (x1 + x2 + x3) / 3.0,
            (y1 + y2 + y3) / 3.0,
            (z1 + z2 + z3) / 3.0,
        )
        face_nodes[key] = nid
        return nid

    # Boundary triangles → 3 quads each (conforming with hex faces on boundary)
    quadrilaterals: list[tuple[int, int, int, int, int]] = []
    for a, b, c, label in mesh.triangles:
        mab = midpoint(a, b)
        mbc = midpoint(b, c)
        mca = midpoint(c, a)
        f = face_center(a, b, c)
        quadrilaterals.append((a, mab, f, mca, label))
        quadrilaterals.append((b, mbc, f, mab, label))
        quadrilaterals.append((c, mca, f, mbc, label))

    # Each tet (a, b, c, d) → 4 hexahedra.
    #
    # Node legend for a single tet:
    #   Original vertices : a, b, c, d
    #   Edge midpoints    : eab, eac, ead, ebc, ebd, ecd
    #   Face centroids    : fabc (opposite d), fabd (opposite c),
    #                       facd (opposite b), fbcd (opposite a)
    #   Tet centroid      : g
    #
    # Each hex is built around one original vertex.  The 8 nodes of the hex
    # associated with vertex V are:
    #   V,  midpoints of the 3 edges from V,
    #   centroids of the 3 faces touching V,
    #   tet centroid g
    #
    # Node ordering follows the Medit/VTK convention for a hexahedron:
    #   bottom face (CCW when viewed from outside) then top face directly above.
    #
    #   hex_a: a  eab  fabc  eac  |  ead  fabd  g  facd
    #   hex_b: b  ebc  fabc  eab  |  ebd  fbcd  g  fabd
    #   hex_c: c  eac  fabc  ebc  |  ecd  facd  g  fbcd   (fixed vs original)
    #   hex_d: d  ead  fabd  ebd  |  ecd  facd  g  fbcd   (fixed vs original)

    hexahedra: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    for a, b, c, d, label in mesh.tetrahedra:
        eab = midpoint(a, b)
        eac = midpoint(a, c)
        ead = midpoint(a, d)
        ebc = midpoint(b, c)
        ebd = midpoint(b, d)
        ecd = midpoint(c, d)

        fabc = face_center(a, b, c)  # face opposite d
        fabd = face_center(a, b, d)  # face opposite c
        facd = face_center(a, c, d)  # face opposite b
        fbcd = face_center(b, c, d)  # face opposite a

        xa, ya, za = coords[a - 1]
        xb, yb, zb = coords[b - 1]
        xc, yc, zc = coords[c - 1]
        xd, yd, zd = coords[d - 1]
        g = add_vertex(
            (xa + xb + xc + xd) / 4.0,
            (ya + yb + yc + yd) / 4.0,
            (za + zb + zc + zd) / 4.0,
        )

        # Hex around vertex a
        hexahedra.append((a, eab, fabc, eac, ead, fabd, g, facd, label))
        # Hex around vertex b
        hexahedra.append((b, ebc, fabc, eab, ebd, fbcd, g, fabd, label))
        # Hex around vertex c  (corrected node ordering)
        hexahedra.append((c, eac, fabc, ebc, ecd, facd, g, fbcd, label))
        # Hex around vertex d  (corrected node ordering)
        hexahedra.append((d, ead, fabd, ebd, ecd, facd, g, fbcd, label))

    vertices_out = [(x, y, z, r) for (x, y, z), r in zip(coords, refs)]
    return vertices_out, quadrilaterals, hexahedra


def write_medit_hex_mesh(
    path: Path,
    vertices: list[tuple[float, float, float, int]],
    quadrilaterals: list[tuple[int, int, int, int, int]],
    hexahedra: list[tuple[int, int, int, int, int, int, int, int, int]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension\n3\n")

        f.write("Vertices\n")
        f.write(f"{len(vertices)}\n")
        for x, y, z, r in vertices:
            f.write(f"{x:.16g} {y:.16g} {z:.16g} {r}\n")

        if quadrilaterals:
            f.write("Quadrilaterals\n")
            f.write(f"{len(quadrilaterals)}\n")
            for a, b, c, d, r in quadrilaterals:
                f.write(f"{a} {b} {c} {d} {r}\n")

        f.write("Hexahedra\n")
        f.write(f"{len(hexahedra)}\n")
        for a, b, c, d, e, f6, g, h, r in hexahedra:
            f.write(f"{a} {b} {c} {d} {e} {f6} {g} {h} {r}\n")

        f.write("End\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Medit tetra mesh to hex-only mesh")
    parser.add_argument("input", type=Path, help="Input .mesh file containing Tetrahedra")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .mesh file")
    args = parser.parse_args()

    mesh = read_medit_mesh(args.input)
    vertices, quads, hexes = convert_tet_to_hex(mesh)
    write_medit_hex_mesh(args.output, vertices, quads, hexes)  # was commented out

    print(f"Input  tetrahedra : {len(mesh.tetrahedra)}")
    print(f"Output hexahedra  : {len(hexes)}  (= {len(mesh.tetrahedra)} × 4)")
    print(f"Output vertices   : {len(vertices)}")
    print(f"Output quads      : {len(quads)}  (= {len(mesh.triangles)} × 3)")


if __name__ == "__main__":
    main()