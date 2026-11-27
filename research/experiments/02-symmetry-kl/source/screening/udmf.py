"""Local UDMF serializer copied from the earlier screening geometry builder."""

from itertools import accumulate

TILE_THINGS = {"z": 3004, "i": 3001, "s": 9, "+": 2012, "a": 2048, "k": 13}

def layout(path, rows, start_angle, x_overrides, y_overrides, official):
    width = len(rows[0])
    assert all(len(row) == width for row in rows)
    assert sum(row.count("P") for row in rows) == sum(row.count("X") for row in rows) == 1
    tiles = {(x, y): cell for y, row in enumerate(rows) for x, cell in enumerate(row) if cell != "#"}
    xs = tuple(accumulate((x_overrides.get(x, 128) for x in range(width)), initial=0))
    ys = tuple(accumulate((y_overrides.get(y, 128) for y in range(len(rows))), initial=0))
    sectors = {tile: i for i, tile in enumerate(tiles)}
    vertices, sides, lines, exits, things = {}, [], [], [], []

    def vertex(point):
        return vertices.setdefault(point, len(vertices))

    def side(sector, middle="-", top="-"):
        sides.append(f'sidedef {{ sector = {sector}; texturemiddle = "{middle}"; '
                     f'texturetop = "{top}"; texturebottom = "-"; }}')
        return len(sides) - 1

    for (x, y), symbol in tiles.items():
        sw, nw, ne, se = (xs[x], ys[y]), (xs[x], ys[y + 1]), (xs[x + 1], ys[y + 1]), (xs[x + 1], ys[y])
        edges = (((x - 1, y), sw, nw), ((x, y + 1), nw, ne), ((x + 1, y), ne, se), ((x, y - 1), se, sw))
        own = sectors[x, y]
        for neighbor, a, b in edges:
            other = sectors.get(neighbor)
            if other is None:
                lines.append(f"linedef {{ v1 = {vertex(a)}; v2 = {vertex(b)}; sidefront = {side(own, 'STARTAN2')}; blocking = true; }}")
                continue
            if other < own:
                continue
            other_symbol = tiles[neighbor]
            front, back = own, other
            special, top = "", "-"
            if "D" in (symbol, other_symbol):
                if symbol == "D":
                    a, b, front, back = b, a, back, front
                special = ("special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = 129; "
                           "playeruse = true; repeatspecial = true;")
                top = "DOORRED"
            if "X" in (symbol, other_symbol):
                if symbol == "X":
                    a, b, front, back = b, a, back, front
                special = "special = 243; playercross = true;"
                exits.append({"linedef": len(lines), "special": 243, "activation": "cross",
                              "destination": "normal", "from_xy": a, "to_xy": b})
            lines.append(f"linedef {{ v1 = {vertex(a)}; v2 = {vertex(b)}; sidefront = {side(front, top=top)}; "
                         f"sideback = {side(back)}; twosided = true; {special} }}")
        kind = 1 if symbol == "P" else TILE_THINGS.get(symbol)
        if kind:
            cx, cy = (xs[x] + xs[x + 1]) // 2, (ys[y] + ys[y + 1]) // 2
            angle = start_angle if kind == 1 else 180
            things.append(f"thing {{ x = {cx}.0; y = {cy}.0; angle = {angle}; type = {kind}; "
                          "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")
    assert exits
    object_text = ['namespace = "zdoom";']
    object_text.extend(f"vertex {{ x = {x}.0; y = {y}.0; }}" for x, y in vertices)
    object_text.extend(f'sector {{ texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1"; '
                       f'heightfloor = 0; heightceiling = {0 if cell == "D" else 128}; lightlevel = 192; }}'
                       for cell in tiles.values())
    object_text.extend([*sides, *lines, *things])
    text = "\n".join(object_text).encode()
    lump = official["Lump"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    official["write_wad"](path, [lump("MAP01", b""), lump("TEXTMAP", text), lump("ENDMAP", b"")])
    return exits


