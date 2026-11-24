"""Descarga fotos de voleibol con licencia libre desde Wikimedia Commons.

Uso:
    python scripts/download_samples.py            # 6 imágenes por defecto
    python scripts/download_samples.py --n 10 --consulta "beach volleyball"

Usa la API pública de Wikimedia Commons, acepta solo licencias libres
(CC0, dominio público, CC BY y CC BY-SA), guarda las imágenes reducidas en
``data/samples/descargadas/`` (carpeta ignorada por git) y escribe
``CREDITOS.md`` con autor, licencia y enlace de cada una, como piden esas licencias.

Si no hay conexión, la app funciona igual con las muestras sintéticas
(``scripts/generar_sinteticos.py``). También puedes copiar tus propias fotos o
videos a ``data/samples/descargadas/``.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volleyvision.utils.io import DIR_MUESTRAS, cargar_imagen, guardar_imagen  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
CABECERAS = {"User-Agent": "VolleyVision/1.0 (proyecto academico de vision artificial)"}
LICENCIAS_OK = ("cc0", "public domain", "pd", "cc by", "cc-by", "cc by-sa", "cc-by-sa")
CONSULTAS = ("volleyball match", "volleyball spike", "volleyball block net", "volleyball women", "volleyball men")


def _limpiar(texto: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", texto or "")).strip()


def buscar(consulta: str, limite: int = 30) -> list[dict]:
    """Busca archivos de imagen en Commons y devuelve su URL y metadatos de licencia."""
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {consulta}",
        "gsrnamespace": 6, "gsrlimit": limite, "prop": "imageinfo", "iiprop": "url|extmetadata|size",
        "iiurlwidth": 1280,
    }
    r = requests.get(API, params=params, headers=CABECERAS, timeout=30)
    r.raise_for_status()
    paginas = r.json().get("query", {}).get("pages", {}).values()
    resultados = []
    for p in paginas:
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        licencia = _limpiar(meta.get("LicenseShortName", {}).get("value", ""))
        resultados.append({
            "titulo": p.get("title", ""), "url": info.get("thumburl") or info.get("url"),
            "pagina": info.get("descriptionurl", ""), "licencia": licencia,
            "autor": _limpiar(meta.get("Artist", {}).get("value", "desconocido")),
            "ancho": info.get("width", 0), "alto": info.get("height", 0),
        })
    return resultados


def licencia_libre(licencia: str) -> bool:
    lic = licencia.lower()
    return any(lic.startswith(x) for x in LICENCIAS_OK) and "nc" not in lic and "nd" not in lic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6, help="número de imágenes a descargar")
    ap.add_argument("--consulta", action="append", help="términos de búsqueda (se puede repetir)")
    args = ap.parse_args()
    destino = DIR_MUESTRAS / "descargadas"
    destino.mkdir(parents=True, exist_ok=True)
    creditos = ["# Créditos de las imágenes descargadas\n",
                "Imágenes de Wikimedia Commons con licencia libre, reducidas a 1280 px.\n"]
    vistos, n = set(), 0
    for consulta in args.consulta or CONSULTAS:
        try:
            candidatos = buscar(consulta)
        except requests.RequestException as e:
            print(f"No se pudo consultar Wikimedia ({e}). Usa las muestras sintéticas.")
            return
        for c in candidatos:
            if n >= args.n:
                break
            if c["titulo"] in vistos or not c["url"] or not licencia_libre(c["licencia"]):
                continue
            if "volley" not in c["titulo"].lower() or "beach" in c["titulo"].lower():
                continue  # solo voleibol de coliseo (el pipeline asume cancha 9 × 18 m)
            if c["ancho"] < 800 or c["ancho"] < c["alto"]:  # solo horizontales y de buena resolución
                continue
            vistos.add(c["titulo"])
            try:
                datos = requests.get(c["url"], headers=CABECERAS, timeout=60).content
                img = cargar_imagen(datos, max_lado=1280)
            except (requests.RequestException, ValueError) as e:
                print(f"  omitida {c['titulo']}: {e}")
                continue
            n += 1
            nombre = f"voleibol_{n:02d}.jpg"
            guardar_imagen(destino / nombre, img)
            creditos.append(f"- **{nombre}**: [{c['titulo']}]({c['pagina']}), autor: {c['autor']}, "
                            f"licencia: {c['licencia']}")
            print(f"  ✓ {nombre} ← {c['titulo']} ({c['licencia']})")
        if n >= args.n:
            break
    (destino / "CREDITOS.md").write_text("\n".join(creditos) + "\n", encoding="utf-8")
    print(f"{n} imágenes guardadas en {destino}")


if __name__ == "__main__":
    main()
