import os, json, yaml
from typing import Dict, List, Any, Union

from jinja2 import Template
import markdown

def generate_report(
    reports_json_path: Union[str, list[Any]],
    stride_yaml_path: str,
    controls_yaml: str,
    conf_threshold: float = 0.2,
    reports_path: str     = "."
)-> str:
    """
    Generate threat model reports in Markdown and HTML formats
    based on detection results, STRIDE mappings, and control mitigations.

    Args:
        reports_json_path (str): Path to the JSON file containing detection outputs.
        stride_yaml_path (str): Path to the YAML file mapping component classes to STRIDE threats.
        controls_yaml_path (str): Path to the YAML file mapping STRIDE threats to CWEs and controls.
        conf_threshold (float, optional): Confidence threshold to filter detections. Defaults to 0.2.
        output_dir (str, optional): Directory where report.md and report.html will be saved. Defaults to current directory.
    """
    # Load detection results and mapping files
    detections = (
        json.load(open(reports_json_path))
        if isinstance(reports_json_path, str)
        else reports_json_path
    )
    stride_map  = yaml.safe_load(open(stride_yaml_path))
    controls_map= yaml.safe_load(open(controls_yaml))

    # Merge components, applying confidence filter and mapping threats/controls
    components = []
    for item in detections:
        img = os.path.basename(item["path"])
        for det in item["boxes"]:
            conf = det.get("confidence", det.get("conf", 0))
            if conf < conf_threshold: continue
            cls = det["name"]
            threats = stride_map.get(cls, [])
            ctrls   = {t: controls_map[t] for t in threats}
            components.append({
                "image": img, "class": cls, "conf": conf,
                "bbox": det.get("coordinates", det.get("xyxy")),
                "threats": threats, "controls": ctrls
            })

    # ender Markdown report using Jinja2 template
    md_tpl = """\
# Threat Model Report

Gerado automaticamente

{% for c in components %}
## {{ c.class }}  ({{ c.image }})
- **Confiança:** {{ '%.2f'|format(c.conf) }}
- **BBox:** {{ c.bbox }}
- **Ameaças:**
  {% for t in c.threats %}
  - **{{ t }}**  
    - CWEs: {{ c.controls[t].cwe }}  
    - Contramedidas:
      {% for m in c.controls[t].controls %}
      - {{ m }}
      {% endfor %}
  {% endfor %}

{% endfor %}
"""
    tpl = Template(md_tpl)
    md = tpl.render(components=components)
    md_path  = os.path.join(reports_path, "report.md")
    open(md_path, "w", encoding="utf-8").write(md)

    # Convert Markdown to HTML
    html_body = markdown.markdown(md, extensions=["tables"])
    html = f"""<!DOCTYPE html>
<html><head><style>body {{background-color: #ffffff !important;color: #000000 !important; }} </style><meta charset="utf-8"><title>Report</title></head><body>{html_body}</body></html>"""
    html_path = os.path.join(reports_path, "report.html")
    open(html_path, "w", encoding="utf-8").write(html)

    print("Reports generated:", reports_path)
    return html


def load_report_results(json_path: str) -> List[Dict[str, Any]]:
    """
    Load detection results from a JSON file.

    Args:
        json_path (str): Path to the JSON file.
    Returns:
        List[Dict[str, Any]]: Parsed detections list.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dictionary.

    Args:
        path (str): Path to the YAML file.
    Returns:
        Dict[str, Any]: Parsed YAML content.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_components(
    detections: List[Dict[str, Any]],
    stride_map: Dict[str, List[str]],
    controls_map: Dict[str, Dict[str, Any]],
    conf_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Merge raw detections with STRIDE and control mappings, filtering by confidence.

    Args:
        detections (List[Dict[str, Any]]): Raw detection entries.
        stride_map (Dict[str, List[str]]): Mapping of classes to STRIDE threats.
        controls_map (Dict[str, Dict[str, Any]]): Mapping of threats to CWE and controls.
        conf_threshold (float, optional): Minimum confidence to include. Defaults to 0.0.
    Returns:
        List[Dict[str, Any]]: Enriched component list.
    """
    components = []
    for item in detections:
        image_path = item.get('path', item.get('file', ''))
        for det in item.get('boxes', []):
            conf = det.get('confidence', det.get('conf', 0))
            if conf < conf_threshold:
                continue
            cls = det.get('name')
            bbox = det.get('coordinates', det.get('xyxy', []))
            threats = stride_map.get(cls, [])
            controls = {t: controls_map.get(t, {}) for t in threats}
            components.append({
                'image': os.path.basename(image_path),
                'class': cls,
                'conf': conf,
                'bbox': bbox,
                'threats': threats,
                'controls': controls
            })
    return components


def render_markdown(
    components: List[Dict[str, Any]],
    template_str: str,
    output_path: str
) -> str:
    """
    Render a Markdown report from a Jinja2 template and component data.

    Args:
        components (List[Dict[str, Any]]): Enriched component list.
        template_str (str): Jinja2 template string for Markdown.
        output_path (str): File path to write the Markdown.
    Returns:
        str: Rendered Markdown string.
    """
    template = Template(template_str)
    md = template.render(components=components)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    return md


def render_html(
    md_content: str,
    output_path: str
) -> str:
    """
    Convert Markdown content to HTML with basic styling.

    Args:
        md_content (str): Raw Markdown text.
        output_path (str): File path to write the HTML.
    Returns:
        str: Full HTML document string.
    """
    # You could use a Markdown-to-HTML conversion
    html_body = markdown.markdown(md_content, extensions=['tables'])
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Threat Model Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5em; }}
        th {{ background: #f5f5f5; text-align: left; }}
        code {{ background: #eee; padding: 2px 4px; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html
