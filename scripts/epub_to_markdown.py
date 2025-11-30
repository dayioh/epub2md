#!/usr/bin/env python3
"""Pipeline minimal pour transformer un EPUB en Markdown autonome."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path


SPAN_ID = re.compile(r'<span[^>]*id="([^"]+)"[^>]*></span>')
HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
FIGURE = re.compile(r'<figure[^>]*>\s*(<img[^>]+>)\s*</figure>', re.MULTILINE)
IMG_WITH_ALT = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*alt="([^"]*)"[^>]*?/?>')
IMG_NO_ALT = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*?/?>')
MARKDOWN_IMAGE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
HTML_SPAN = re.compile(r'<span[^>]*>(.*?)</span>', re.DOTALL)
HTML_EMPTY_SPAN = re.compile(r'<span[^>]*></span>')
HTML_LINK = re.compile(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
HTML_CODE_BLOCK = re.compile(r'<pre>\s*<code([^>]*)>(.*?)</code>\s*</pre>', re.DOTALL | re.IGNORECASE)
INTERNAL_LINK = re.compile(r'\[([^\]]+)\]\(#([^)]+)\)')
WIKILINK = re.compile(r'\[\[([^\]#]+)#([^|\]]+)(?:\|([^\]]+))?\]\]')
TOC_SPACES = re.compile(r'^(\s*\d+)\.\s{2,}', re.MULTILINE)
LIST_LINE_PATTERN = re.compile(r'^\s*(?:\d+\.\s|[-*+]\s)')
NAV_PATTERN = re.compile(r'<nav[^>]*>.*?</nav>', re.DOTALL | re.IGNORECASE)
ASIDE_PATTERN = re.compile(r'<aside[^>]*>(.*?)</aside>', re.DOTALL | re.IGNORECASE)
FIGCAPTION_PATTERN = re.compile(r'<figcaption[^>]*>(.*?)</figcaption>', re.DOTALL | re.IGNORECASE)
LIST_LINE_PATTERN = re.compile(r'^\s*(?:\d+\.\s|[-*+]\s)')


def parse_metadata_block(block: str) -> dict[str, list[str]]:
    metadata: dict[str, list[str]] = {}
    for raw_line in block.splitlines():
        if ':' not in raw_line:
            continue
        key, value = raw_line.split(':', 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key and value:
            metadata.setdefault(key, []).append(value)
    return metadata


def slugify(text: str) -> str:
    """Génère un slug GitHub-like pour les ancres."""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text.strip())
    text = re.sub(r'-{2,}', '-', text)
    return text or 'section'


def split_front_matter(markdown: str) -> tuple[dict[str, list[str]], str]:
    if not markdown.startswith('---'):
        return {}, markdown
    try:
        _, block, body = markdown.split('---', 2)
    except ValueError:
        return {}, markdown
    return parse_metadata_block(block), body.lstrip('\n')


def build_anchor_map(body: str) -> dict[str, str]:
    anchor_map: dict[str, str] = {}
    pending_ids: list[str] = []
    seen: dict[str, int] = {}

    for line in body.splitlines():
        pending_ids.extend(SPAN_ID.findall(line))
        match = HEADING.match(line)
        if not match:
            continue
        text = SPAN_ID.sub('', match.group(2))
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            text = 'Section'
        key = text.lower()
        index = seen.get(key, 0)
        seen[key] = index + 1
        heading_text = text if index == 0 else f'{text} ({index + 1})'
        for anchor_id in pending_ids:
            anchor_map[anchor_id] = heading_text
        pending_ids.clear()
    return anchor_map


def clean_html(body: str) -> str:
    """Supprime les balises non désirées et convertit les <img> en markdown."""

    def replace_link(match: re.Match[str]) -> str:
        href = match.group(1).strip()
        text = match.group(2)
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text.strip()) or href
        return f'[{text}]({href})'

    def replace_code_block(match: re.Match[str]) -> str:
        attrs, code = match.groups()
        language = ''
        if attrs:
            class_match = re.search(r'class="([^"]+)"', attrs)
            if class_match:
                classes = class_match.group(1).split()
                for cls in classes:
                    if cls.startswith('language-'):
                        language = cls.split('language-')[-1]
                        break
                if not language and classes:
                    language = classes[0]
        # Normalise code text
        code_text = html.unescape(code)
        code_text = code_text.replace('\r\n', '\n').replace('\r', '\n')
        code_text = re.sub(r'<br\s*/?>', '\n', code_text)
        code_text = re.sub(r'</?[^>]+>', '', code_text)
        code_text = code_text.strip('\n')
        fence = f'```{language}'.rstrip()
        return f'{fence}\n{code_text}\n```'

    def replace_aside(match: re.Match[str]) -> str:
        content = match.group(1)
        content = content.replace('<em>', '*').replace('</em>', '*')
        content = content.replace('<strong>', '**').replace('</strong>', '**')
        content = re.sub(r'</?[^>]+>', '', content)
        text = html.unescape(content)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(f'> {line}' for line in lines)

    def replace_figcaption(match: re.Match[str]) -> str:
        text = match.group(1)
        text = html.unescape(re.sub(r'</?[^>]+>', '', text)).strip()
        return f'*{text}*' if text else ''

    body = NAV_PATTERN.sub('', body)
    body = ASIDE_PATTERN.sub(replace_aside, body)
    body = FIGCAPTION_PATTERN.sub(replace_figcaption, body)
    body = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)
    body = re.sub(r'</?sup[^>]*>', '', body)
    body = re.sub(r'</?nav[^>]*>', '', body)
    body = HTML_LINK.sub(replace_link, body)
    body = HTML_CODE_BLOCK.sub(replace_code_block, body)
    def to_markdown(img_html: str) -> str:
        return IMG_WITH_ALT.sub(
            lambda m: f'![{m.group(2).strip() or "Image"}]({m.group(1)})',
            img_html,
        )

    body = FIGURE.sub(lambda m: to_markdown(m.group(1)), body)
    body = IMG_WITH_ALT.sub(
        lambda m: f'![{m.group(2).strip() or "Image"}]({m.group(1)})', body
    )
    body = IMG_NO_ALT.sub(lambda m: f'![Image]({m.group(1)})', body)
    body = re.sub(r'</?(article|div|section|figure)[^>]*>', '', body)
    body = HTML_EMPTY_SPAN.sub('', body)
    body = HTML_SPAN.sub(r'\1', body)
    body = re.sub(r'<br\s*/?>', '\n\n', body)
    body = re.sub(r'&nbsp;', ' ', body)
    body = re.sub(r'\n{3,}', '\n\n', body)
    return body


@dataclass(frozen=True)
class ImageOptions:
    quality: int
    max_dimension: int | None = None
    target_kb: int | None = None
    strip_metadata: bool = True


def encode_image(image_path: Path, options: ImageOptions) -> tuple[str, str]:
    """Retourne (base64, mime) en recompressant en WebP avec quelques garde-fous."""
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    try:
        cmd = ['magick', str(image_path)]
        if options.strip_metadata:
            cmd.append('-strip')
        if options.max_dimension:
            geometry = f'{options.max_dimension}x{options.max_dimension}>'
            cmd.extend(['-resize', geometry])
        cmd.extend(['-quality', str(options.quality)])
        if options.target_kb:
            target_bytes = max(options.target_kb, 1) * 1024
            cmd.extend(['-define', f'webp:target-size={target_bytes}'])
        cmd.append('webp:-')
        result = subprocess.run(cmd, check=True, capture_output=True)
        data = result.stdout
        mime = 'image/webp'
    except (subprocess.CalledProcessError, FileNotFoundError):
        data = image_path.read_bytes()
        mime = mimetypes.guess_type(image_path.name)[0] or 'application/octet-stream'
    return base64.b64encode(data).decode('ascii'), mime


def embed_images(markdown: str, media_root: Path, options: ImageOptions) -> str:
    """Remplace les chemins locaux par des data URI."""
    cache: dict[Path, tuple[str, str]] = {}

    def repl(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        if not src.startswith(str(media_root)):
            return match.group(0)
        img_path = Path(src)
        cached = cache.get(img_path)
        if not cached:
            try:
                cached = encode_image(img_path, options)
            except FileNotFoundError:
                return match.group(0)
            cache[img_path] = cached
        data, mime = cached
        return f'![{alt}](data:{mime};base64,{data})'

    return MARKDOWN_IMAGE.sub(repl, markdown)


def rewrite_links(markdown: str, id_to_slug: dict[str, str], target_name: str) -> str:
    """Remplace les ancres internes par des wikilinks [[fichier#titre]]."""

    def repl(match: re.Match[str]) -> str:
        label, target = match.groups()
        slug_text = id_to_slug.get(target) or label.strip()
        return f'[[{target_name}#{slug_text}|{label}]]'

    return INTERNAL_LINK.sub(repl, markdown)


def fix_toc_spacing(markdown: str) -> str:
    """Nettoie les listes numérotées pour un rendu compact."""
    markdown = TOC_SPACES.sub(lambda m: f'{m.group(1)}. ', markdown)

    def is_numbered(line: str) -> bool:
        return bool(re.match(r'\s*\d+\.\s', line))

    lines = markdown.splitlines()
    cleaned: list[str] = []

    for i, line in enumerate(lines):
        if (
            line.strip() == ''
            and i > 0
            and i + 1 < len(lines)
            and is_numbered(lines[i - 1])
            and is_numbered(lines[i + 1])
        ):
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


def is_list_line(line: str) -> bool:
    return bool(LIST_LINE_PATTERN.match(line))


def compact_lists(markdown: str) -> str:
    """Uniformise les listes (espaces et lignes vides superflus)."""
    lines = markdown.splitlines()
    cleaned: list[str] = []

    for idx, original_line in enumerate(lines):
        line = re.sub(r'^(\s*\d+\.)\s+', r'\1 ', original_line)
        line = re.sub(r'^(\s*[-*+])\s+', r'\1 ', line)
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ''
        if (
            line.strip() == ''
            and cleaned
            and idx + 1 < len(lines)
            and is_list_line(cleaned[-1])
            and is_list_line(next_line)
        ):
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


def normalize_wikilinks(markdown: str) -> str:
    """Nettoie les wikilinks (espaces superflus)."""

    def repl(match: re.Match[str]) -> str:
        file_name, slug, label = match.groups()
        label = (label or slug).strip()
        slug_text = slug.strip()
        return f'[[{file_name}#{slug_text}|{label}]]'

    return WIKILINK.sub(repl, markdown)


def generate_table_of_contents(markdown: str, target_name: str) -> str:
    """Génère une table des matières Markdown à partir des titres de niveau 2."""
    lines = markdown.splitlines()
    headings: list[str] = []
    for line in lines:
        if line.startswith('## ') and not line.startswith('## Table of Contents'):
            title = line[3:].strip()
            if title:
                headings.append(title)
    if not headings:
        return markdown
    toc_lines = ['## Table of Contents', '']
    toc_lines.extend(
        f'{idx}. [[{target_name}#{title}|{title}]]'
        for idx, title in enumerate(headings, start=1)
    )
    toc_lines.append('')
    try:
        start = lines.index('## Table of Contents')
    except ValueError:
        # Insère le toc au début après metadata
        return '\n'.join(toc_lines + [''] + lines)
    end = start + 1
    while end < len(lines) and not lines[end].startswith('## '):
        end += 1
    new_lines = lines[:start] + toc_lines + lines[end:]
    return '\n'.join(new_lines)


def merge_chapter_headings(markdown: str) -> str:
    """Fusionne les couples 'Chapter X' + titre qui suivent immédiatement."""
    lines = markdown.splitlines()
    merged: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith('## Chapter '):
            # Cherche le prochain heading de même niveau
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and lines[j].startswith('## '):
                title1 = line[3:].strip()
                title2 = lines[j][3:].strip()
                if not title2.lower().startswith('chapter '):
                    merged.append(f'## {title1} {title2}')
                    i = j + 1
                    continue
        merged.append(line)
        i += 1

    return '\n'.join(merged)


def build_meta_block(metadata: dict[str, list[str]]) -> str:
    """Formate l’entête Markdown."""
    lines = []
    for key in ('title', 'author', 'language', 'publisher', 'date'):
        if metadata.get(key):
            label = key.capitalize() + 's' if key == 'author' else key.capitalize()
            lines.append(f'**{label}:** {metadata[key][0]}')

    for value in metadata.get('identifier', []):
        cleaned = value.strip('- ').strip()
        if cleaned.lower().startswith('isbn'):
            lines.append(f'**Identifier:** {cleaned}')
            break
    return '\n'.join(lines)


def convert_epub(
    epub_path: Path,
    workdir: Path,
    output_path: Path,
    image_options: ImageOptions,
) -> None:
    """Pipeline complet de conversion."""
    workdir.mkdir(parents=True, exist_ok=True)
    raw_path = workdir / 'raw.md'
    media_dir = workdir / 'media'
    if media_dir.exists():
        shutil.rmtree(media_dir)

    pandoc_cmd = [
        'pandoc',
        str(epub_path),
        '--from=epub',
        '--to=gfm',
        '--standalone',
        '--wrap=none',
        '--markdown-headings=atx',
        f'--extract-media={media_dir}',
        '-o',
        str(raw_path),
    ]
    subprocess.run(pandoc_cmd, check=True)

    raw_text = raw_path.read_text(encoding='utf-8')
    metadata, body = split_front_matter(raw_text)
    id_to_slug = build_anchor_map(body)
    body = clean_html(body)
    body = embed_images(body, media_dir, image_options)
    body = rewrite_links(body, id_to_slug, output_path.name)
    body = compact_lists(body)
    body = fix_toc_spacing(body)
    body = merge_chapter_headings(body)
    body = normalize_wikilinks(body)
    body = generate_table_of_contents(body, output_path.name)
    final_text = build_meta_block(metadata).strip() + '\n\n' + body.lstrip('\n')
    output_path.write_text(final_text, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('epub', type=Path, help='Chemin du fichier EPUB source')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('output.md'),
        help='Fichier Markdown de destination',
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        default=Path('.epub-workdir'),
        help='Dossier temporaire pour les ressources intermédiaires',
    )
    parser.add_argument(
        '--webp-quality',
        type=int,
        default=60,
        help='Qualité WebP (0-100). 60 offre un bon compromis sous 10Mo.',
    )
    parser.add_argument(
        '--max-dimension',
        type=int,
        default=1400,
        help='Taille maximale (px) appliquée seulement si l’image est plus grande.',
    )
    parser.add_argument(
        '--target-kb',
        type=int,
        default=6,
        help='Poids cible approximatif par image (en kilo-octets).',
    )
    parser.add_argument(
        '--keep-metadata',
        action='store_true',
        help='Conserver les métadonnées embarquées dans les images.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_options = ImageOptions(
        quality=args.webp_quality,
        max_dimension=args.max_dimension,
        target_kb=args.target_kb,
        strip_metadata=not args.keep_metadata,
    )
    convert_epub(args.epub, args.work_dir, args.output, image_options)


if __name__ == '__main__':
    main()

