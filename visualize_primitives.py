#!/usr/bin/env python3
"""
各プリミティブの可視化を生成・保存するスクリプト
リファクタリング後の新しい実装に対応
"""

import os

import numpy as np
import plotly.graph_objects as go
import torch

# 基本的なプリミティブ
from src.fdslxsdf4seg.basic_sdf import (
    ConcaveCylinder,
    Cone,
    ConeCylinder,
    ConvexCylinder,
    Cylinder,
    Octahedron,
    Sphere,
    Torus,
)
from src.fdslxsdf4seg.generate_sdf_dataset import (
    visualize_sample,
)

# Onioned プリミティブ
from src.fdslxsdf4seg.onioned_prism.onioned_sector_polygon_prism import (
    OnionedCylinder,
    OnionedHexagonPrism,
    OnionedPentagonPrism,
    OnionedSquareConcavePrism,
    OnionedSquareConePrism,
    OnionedSquareConvexPrism,
    OnionedSquarePrism,
    OnionedTriangleConcavePrism,
    OnionedTriangleConePrism,
    OnionedTriangleConvexPrism,
    OnionedTrianglePrism,
)

# Onioned Star プリミティブ
from src.fdslxsdf4seg.onioned_prism.onioned_star_polygon_prism import (
    OnionedEightStarConcavePrism,
    OnionedEightStarConePrism,
    OnionedEightStarConvexPrism,
    OnionedEightStarPrism,
    OnionedFiveStarConcavePrism,
    OnionedFiveStarConePrism,
    OnionedFiveStarConvexPrism,
    OnionedFiveStarPrism,
    OnionedSevenStarConcavePrism,
    OnionedSevenStarConePrism,
    OnionedSevenStarConvexPrism,
    OnionedSevenStarPrism,
    OnionedSixStarConcavePrism,
    OnionedSixStarConePrism,
    OnionedSixStarConvexPrism,
    OnionedSixStarPrism,
)

# Revolution系
from src.fdslxsdf4seg.revolution.star_revolution import (
    FiveStarRevolution,
    FourStarRevolution,
    ThreeStarRevolution,
)

# SDF Object ベースクラス
from src.fdslxsdf4seg.sdf_object import (
    SectorPolygonPrism,
)

# 凹セクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.concave_sector_polygon_prism import (
    SquareConcavePrism,
    TriangleConcavePrism,
)

# コーンセクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.cone_sector_polygon_prism import (
    HexagonConePrism,
    PentagonConePrism,
    SquareConePrism,
    TriangleConePrism,
)

# 凸セクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.convex_sector_polygon_prism import (
    SquareConvexPrism,
    TriangleConvexPrism,
)

# セクターポリゴンプリズム
from src.fdslxsdf4seg.sector_polygon_prism.sector_polygon_prism import (
    HeptagonPrism,
    HexagonPrism,
    NonagonPrism,
    OctagonPrism,
    PentagonPrism,
    SquarePrism,
    TrianglePrism,
)

# スタープリズム
from src.fdslxsdf4seg.star_polygon_prism.star_prism import (
    FiveStarPrism,
    SixStarPrism,
)

# 凸スタープリズム
# 凹スタープリズム
# コーンスタープリズム
# ピラミッドスタープリズム
# トーラス系
from src.fdslxsdf4seg.torus.sector_polygon_torus import (
    HeptagonTorus,
    HexagonTorus,
    NonagonTorus,
    OctagonTorus,
    PentagonTorus,
    SquareTorus,
)
from src.fdslxsdf4seg.torus.star_torus import (
    EightStarTorus,
    FiveStarTorus,
    SevenStarTorus,
    SixStarTorus,
)

# Union プリミティブ
from src.fdslxsdf4seg.union.sphere_tube import (
    FiveStarRevolutionCylinderUnion,
    FiveStarRevolutionPentagonUnion,
    FiveStarRevolutionSquareUnion,
    FiveStarRevolutionTriangleUnion,
    FourStarRevolutionCylinderUnion,
    FourStarRevolutionPentagonUnion,
    FourStarRevolutionSquareUnion,
    FourStarRevolutionTriangleUnion,
    SphereCylinderUnion,
    SpherePentagonUnion,
    SphereSquareUnion,
    SphereTriangleUnion,
    ThreeStarRevolutionCylinderUnion,
    ThreeStarRevolutionPentagonUnion,
    ThreeStarRevolutionSquareUnion,
    ThreeStarRevolutionTriangleUnion,
)


def visualize_primitive_3d_isosurface(sdf_data, output_file, name="Primitive"):
    """
    SDFデータから3D isosurface plotを生成し、半透明で可視化

    Args:
        sdf_data: SDFデータ (torch.Tensor or numpy.ndarray)
        output_file: 出力ファイルパス (HTML形式で保存)
        name: プリミティブ名
    """
    try:
        # NumPy配列に変換
        if isinstance(sdf_data, torch.Tensor):
            sdf_np = sdf_data.cpu().numpy()
        else:
            sdf_np = sdf_data

        # グリッドサイズを取得
        nz, ny, nx = sdf_np.shape

        # 座標メッシュを作成
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        z = np.linspace(0, nz - 1, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # SDFデータの統計情報を取得
        sdf_min = np.min(sdf_np)
        sdf_max = np.max(sdf_np)
        sdf_mean = np.mean(sdf_np)

        print(
            f"    SDF stats: min={sdf_min:.3f}, max={sdf_max:.3f}, mean={sdf_mean:.3f}"
        )

        # isosurface用にSDFデータを調整（0レベルの等値面を表示）
        fig = go.Figure()

        # 適切な等値面範囲を設定
        iso_range = (
            max(abs(sdf_min), abs(sdf_max)) if sdf_max > abs(sdf_min) else abs(sdf_min)
        )
        iso_min = max(sdf_min, -iso_range * 0.8)
        iso_max = min(sdf_max, iso_range * 0.8)

        # メインのisosurface（SDF = 0の等値面を中心に）
        fig.add_trace(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=sdf_np.flatten(),
                isomin=iso_min,
                isomax=iso_max,
                surface_count=5,  # より多くの等値面を表示
                colorscale="RdYlBu",
                opacity=0.7,  # 少し濃くして見やすく
                name=f"{name} Surface",
                showscale=True,
                colorbar=dict(title="SDF Value"),
                caps=dict(x_show=False, y_show=False, z_show=False),  # キャップを非表示
            )
        )

        # レイアウトを設定
        fig.update_layout(
            title=f"3D Isosurface Visualization: {name}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",  # 等尺で表示
                bgcolor="rgba(0,0,0,0)",  # 背景を透明に
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)  # カメラ位置を調整
                ),
            ),
            width=800,
            height=600,
        )

        # HTMLファイルとして保存
        fig.write_html(output_file)
        print(f"  3D Isosurface saved: {output_file}")

        # 静的画像も保存（要kaleido）
        try:
            png_file = output_file.replace(".html", ".png")
            fig.write_image(png_file, width=800, height=600)
            print(f"  3D Isosurface PNG saved: {png_file}")
            return png_file  # PNG ファイルパスを返す
        except Exception as e:
            print(
                f"  Note: Could not save PNG (install kaleido for static images): {e}"
            )
            return None

    except Exception as e:
        print(f"  Error creating 3D isosurface for {name}: {e}")


def visualize_primitive_marching_cubes(sdf_data, output_file, name="Primitive"):
    """
    Marching Cubesアルゴリズムを使用してメッシュを生成し、3D可視化

    Args:
        sdf_data: SDFデータ (torch.Tensor or numpy.ndarray)
        output_file: 出力ファイルパス
        name: プリミティブ名
    """
    try:
        # NumPy配列に変換
        if isinstance(sdf_data, torch.Tensor):
            sdf_np = sdf_data.cpu().numpy()
        else:
            sdf_np = sdf_data

        # Marching Cubesでメッシュを生成（SDF = 0の等値面）
        try:
            import skimage.measure as measure

            verts, faces, normals, values = measure.marching_cubes(sdf_np, level=0.0)
        except ImportError:
            print("  Warning: scikit-image not available, using basic isosurface")
            visualize_primitive_3d_isosurface(sdf_data, output_file, name)
            return

        # Plotlyでメッシュを可視化
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    intensity=np.linspace(0, 1, len(verts)),
                    colorscale="Viridis",
                    opacity=0.7,  # 半透明
                    name=f"{name} Mesh",
                )
            ]
        )

        # レイアウトを設定
        fig.update_layout(
            title=f"3D Mesh Visualization: {name}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
                bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=800,
            height=600,
        )

        # HTMLファイルとして保存
        fig.write_html(output_file)
        print(f"  3D Mesh saved: {output_file}")

        # 静的画像も保存
        try:
            png_file = output_file.replace(".html", ".png")
            fig.write_image(png_file, width=800, height=600)
            print(f"  3D Mesh PNG saved: {png_file}")
            return png_file  # PNG ファイルパスを返す
        except Exception as e:
            print(
                f"  Note: Could not save PNG (install kaleido for static images): {e}"
            )
            return None

    except Exception as e:
        print(f"  Error creating 3D mesh for {name}: {e}")


def extract_plotly_data_from_html(html_file):
    """
    HTMLファイルからPlotlyのデータとレイアウト情報を抽出

    Args:
        html_file: 元のHTMLファイルパス

    Returns:
        tuple: (data, layout) またはNoneのタプル
    """
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Plotly.newPlot()の呼び出しを探す
        import re

        # パターン1: Plotly.newPlot("div-id", data, layout)
        pattern1 = r'Plotly\.newPlot\(["\']([^"\']+)["\'],\s*(\[.*?\]),\s*(\{.*?\})\)'
        match1 = re.search(pattern1, content, re.DOTALL)

        if match1:
            div_id, data_str, layout_str = match1.groups()
            return data_str, layout_str, div_id

        # パターン2: より柔軟なパターン検索
        pattern2 = r'Plotly\.newPlot\(["\']([^"\']+)["\'],\s*(\[[\s\S]*?\]),\s*(\{[\s\S]*?\}),\s*\{.*?\}\)'
        match2 = re.search(pattern2, content, re.DOTALL)

        if match2:
            div_id, data_str, layout_str = match2.groups()
            return data_str, layout_str, div_id

        # パターン3: より簡単なパターン（改行込み）
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "Plotly.newPlot" in line:
                # 次の数行を結合して完全なコードを取得
                code_block = ""
                for j in range(i, min(i + 50, len(lines))):
                    code_block += lines[j] + "\n"
                    if "}" in lines[j] and code_block.count("{") <= code_block.count(
                        "}"
                    ):
                        break

                # 最後の手段として基本的な抽出を試行
                if '"data"' in code_block or "[{" in code_block:
                    return None, None, "plot_extracted"  # プレースホルダー

        return None, None, None

    except Exception as e:
        print(f"    Error extracting Plotly data from {html_file}: {e}")
        return None, None, None


def combine_3d_visualizations(
    primitive_names, output_dir="visualize_output", viz_type="3d"
):
    """
    複数のプリミティブの3D可視化（HTML）を統合したHTMLページを作成し、
    同時に3D PNG画像を並べて一つの画像にも結合する

    Args:
        primitive_names: 結合するプリミティブ名のリスト
        output_dir: 可視化画像が保存されているディレクトリ
        viz_type: 可視化タイプ（"3d" for isosurface, "mesh" for marching cubes）

    Returns:
        dict: 結果のパス（"html": HTMLファイルパス, "png": PNG結合ファイルパス）、None（失敗時）
    """
    try:
        # 結合用のディレクトリを作成
        combined_dir = os.path.join(output_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        # 各プリミティブの3D可視化HTMLファイルとPNGファイルパスを構築
        html_files = []
        png_files = []
        valid_primitives = []

        html_suffix = "_3d.html" if viz_type == "3d" else "_mesh.html"
        png_suffix = "_3d.png" if viz_type == "3d" else "_mesh.png"

        for primitive_name in primitive_names:
            base_name = f"{primitive_name.lower()}_visualization"
            html_path = os.path.join(output_dir, base_name + html_suffix)
            png_path = os.path.join(output_dir, base_name + png_suffix)

            if os.path.exists(html_path):
                html_files.append(html_path)
                valid_primitives.append(primitive_name)

                # PNGファイルも存在する場合は追加
                if os.path.exists(png_path):
                    png_files.append(png_path)
                else:
                    # PNGファイルがない場合はNoneを追加（位置を保持）
                    png_files.append(None)
                    print(
                        f"  Note: PNG file not found for {primitive_name}: {png_path}"
                    )
            else:
                print(
                    f"  Warning: 3D visualization not found for {primitive_name}: {html_path}"
                )

        if not html_files:
            print(f"  Error: No valid {viz_type} visualization files found")
            return None

        # タイムスタンプを生成
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 統合HTMLページを作成（実用的なアプローチ）
        viz_title = "3D Isosurface" if viz_type == "3d" else "3D Mesh"
        output_path = os.path.join(
            combined_dir, f"combined_{viz_type}_primitives_{timestamp}.html"
        )

        # HTMLテンプレートを作成（iframeベースのアプローチ）
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Combined {viz_title} Visualizations</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .primitive-container {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            min-height: 600px;
        }}
        .primitive-title {{
            text-align: center;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .iframe-container {{
            width: 100%;
            height: 500px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .link-button {{
            display: inline-block;
            margin: 5px;
            padding: 8px 16px;
            background-color: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
        }}
        .link-button:hover {{
            background-color: #218838;
        }}
        .fullscreen-note {{
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Combined {viz_title} Visualizations</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Primitives: {len(valid_primitives)} items</p>
    </div>
    
    <div class="controls">
        <h3>Individual Files (Full Interactive Features):</h3>
"""

        # 個別ファイルのリンクボタンを追加
        for primitive_name in valid_primitives:
            base_name = f"{primitive_name.lower()}_visualization"
            suffix = "_3d.html" if viz_type == "3d" else "_mesh.html"
            html_content += f'<a href="../{base_name}{suffix}" target="_blank" class="link-button">{primitive_name} {viz_title}</a>\n'

        html_content += """
    </div>
    
    <div class="grid-container">
"""

        # 各プリミティブのiframeベースの埋め込み
        for i, (html_file, primitive_name) in enumerate(
            zip(html_files, valid_primitives)
        ):
            try:
                base_name = f"{primitive_name.lower()}_visualization"
                suffix = "_3d.html" if viz_type == "3d" else "_mesh.html"
                iframe_src = f"../{base_name}{suffix}"

                html_content += f"""
        <div class="primitive-container">
            <div class="primitive-title">{primitive_name}</div>
            <div class="iframe-container">
                <iframe src="{iframe_src}"
                        title="{primitive_name} {viz_title}"
                        loading="eager">
                    <p>Your browser does not support iframes.
                    <a href="{iframe_src}" target="_blank">Open {primitive_name} {viz_title} in new window</a></p>
                </iframe>
            </div>
        </div>
"""

            except Exception as e:
                print(f"    Warning: Could not process 3D file {html_file}: {e}")
                continue

        html_content += """
    </div>
    
    <div class="fullscreen-note">
        <h3>💡 Usage Tips:</h3>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>Interactive Features:</strong> Each visualization above shows a live preview</li>
            <li><strong>Full Control:</strong> Click the green buttons above for full interactive features</li>
            <li><strong>Better Performance:</strong> Individual files load faster and have more features</li>
            <li><strong>Comparison:</strong> Use this page to quickly compare different primitives</li>
        </ul>
    </div>
    
    <script>
        // ページ読み込み時の初期化
        window.addEventListener('load', function() {
            console.log('Combined 3D visualization loaded successfully');
            
            // iframeの読み込み状況を監視
            const iframes = document.querySelectorAll('iframe');
            let loadedCount = 0;
            
            iframes.forEach((iframe, index) => {
                iframe.addEventListener('load', function() {
                    loadedCount++;
                    console.log(`Iframe ${index + 1}/${iframes.length} loaded`);
                    
                    if (loadedCount === iframes.length) {
                        console.log('All 3D visualizations loaded');
                        // すべてのiframeが読み込まれた後の処理があればここに追加
                    }
                });
                
                iframe.addEventListener('error', function() {
                    console.warn(`Failed to load iframe ${index + 1}: ${iframe.src}`);
                });
            });
        });
    </script>
    
    <div style="text-align: center; margin-top: 30px; padding: 15px; background-color: #e7f3ff; border-radius: 10px;">
        <p><strong>Technical Note:</strong> This combined view uses iframes to embed individual 3D visualizations.</p>
        <p>Each primitive maintains its full interactive capabilities including:</p>
        <p>🔄 <strong>Rotation & Zoom</strong> | 🎨 <strong>Color Controls</strong> | 📊 <strong>Data Inspection</strong> | 💾 <strong>Export Options</strong></p>
    </div>
</body>
</html>
"""

        # HTMLファイルを保存
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"  Combined {viz_title} HTML visualization saved: {output_path}")

        # 結果を初期化
        results = {"html": output_path}

        # PNG画像の結合も実行
        png_combined_path = None
        available_png_files = [path for path in png_files if path is not None]

        if available_png_files:
            print(f"  Combining {len(available_png_files)} PNG images...")
            png_combined_path = combine_3d_png_images(
                available_png_files, valid_primitives, combined_dir, viz_type, timestamp
            )
            if png_combined_path:
                results["png"] = png_combined_path
                print(
                    f"  Combined {viz_title} PNG visualization saved: {png_combined_path}"
                )
        else:
            print("  No PNG files available for combination")

        return results

    except Exception as e:
        print(f"  Error combining {viz_type} visualizations: {e}")
        return None


def combine_3d_png_images(png_files, primitive_names, output_dir, viz_type, timestamp):
    """
    複数の3D PNG画像を並べて一つの画像に結合する

    Args:
        png_files: 結合するPNGファイルパスのリスト
        primitive_names: プリミティブ名のリスト（画像と対応）
        output_dir: 出力ディレクトリ
        viz_type: 可視化タイプ（"3d" or "mesh"）
        timestamp: タイムスタンプ

    Returns:
        str: 結合したPNG画像のパス（成功時）、None（失敗時）
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        if not png_files:
            return None

        # 存在する画像ファイルのみをフィルタリング
        valid_files = []
        valid_names = []
        for i, png_file in enumerate(png_files):
            if png_file and os.path.exists(png_file):
                valid_files.append(png_file)
                valid_names.append(primitive_names[i])

        if not valid_files:
            print("  No valid PNG files found for combination")
            return None

        # 最初の画像を読み込んでサイズを取得
        sample_img = Image.open(valid_files[0])
        img_width, img_height = sample_img.size
        sample_img.close()

        # グリッド設定
        grid_cols = 3
        num_images = len(valid_files)
        grid_rows = (num_images + grid_cols - 1) // grid_cols

        # レイアウト計算
        padding = 20
        title_height = 80
        label_height = 40
        combined_width = grid_cols * img_width + (grid_cols + 1) * padding
        combined_height = (
            grid_rows * (img_height + label_height)
            + (grid_rows + 1) * padding
            + title_height
        )

        # 結合画像を作成
        combined_img = Image.new("RGB", (combined_width, combined_height), "white")
        draw = ImageDraw.Draw(combined_img)

        # フォント設定
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # タイトルを描画
        viz_title = "3D Isosurface" if viz_type == "3d" else "3D Mesh"
        title_text = f"Combined {viz_title} Visualizations ({num_images} primitives)"
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = bbox[2] - bbox[0]
        title_x = (combined_width - title_width) // 2
        draw.text((title_x, 20), title_text, fill="black", font=title_font)

        # 各画像を配置
        for i, (png_file, name) in enumerate(zip(valid_files, valid_names)):
            try:
                # 画像を読み込み
                img = Image.open(png_file)

                # 位置計算
                row = i // grid_cols
                col = i % grid_cols

                x = padding + col * (img_width + padding)
                y = title_height + padding + row * (img_height + label_height + padding)

                # 画像を貼り付け
                combined_img.paste(img, (x, y))
                img.close()

                # ラベルを描画
                label_y = y + img_height + 5
                bbox = draw.textbbox((0, 0), name, font=label_font)
                label_width = bbox[2] - bbox[0]
                label_x = x + (img_width - label_width) // 2
                draw.text((label_x, label_y), name, fill="black", font=label_font)

            except Exception as e:
                print(f"  Warning: Could not process {png_file}: {e}")
                continue

        # 結合画像を保存
        output_path = os.path.join(
            output_dir, f"combined_{viz_type}_png_{timestamp}.png"
        )
        combined_img.save(output_path, "PNG", quality=95)
        combined_img.close()

        return output_path

    except ImportError:
        print("  Error: PIL (Pillow) not available, cannot combine PNG images")
        return None
    except Exception as e:
        print(f"  Error combining PNG images: {e}")
        return None


def combine_primitive_visualizations(
    primitive_names, output_dir="visualize_output", grid_cols=3, include_3d=False
):
    """
    選択した複数のプリミティブの可視化結果を並べて1枚の画像として出力

    Args:
        primitive_names: 結合するプリミティブ名のリスト
        output_dir: 可視化画像が保存されているディレクトリ
        grid_cols: グリッドの列数
        include_3d: 3D可視化も結合するかどうか

    Returns:
        dict: 結合結果（2D画像パス、3D HTMLパスなど）
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # 結合用のディレクトリを作成
        combined_dir = os.path.join(output_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        # 各プリミティブの可視化画像パスを構築
        image_paths = []
        valid_primitives = []

        for primitive_name in primitive_names:
            image_path = os.path.join(
                output_dir, f"{primitive_name.lower()}_visualization.png"
            )
            if os.path.exists(image_path):
                image_paths.append(image_path)
                valid_primitives.append(primitive_name)
            else:
                print(
                    f"  Warning: Visualization image not found for {primitive_name}: {image_path}"
                )

        if not image_paths:
            print("  Error: No valid visualization images found")
            return None

        # 最初の画像を読み込んでサイズを取得
        sample_img = Image.open(image_paths[0])
        img_width, img_height = sample_img.size
        sample_img.close()

        # グリッドサイズを計算
        num_images = len(image_paths)
        grid_rows = (num_images + grid_cols - 1) // grid_cols

        # タイトル用の余白
        title_height = 80
        padding = 15

        # 結合後の画像サイズ
        combined_width = grid_cols * img_width + (grid_cols + 1) * padding
        combined_height = (
            grid_rows * img_height + (grid_rows + 1) * padding + title_height
        )

        # 白い背景の新しい画像を作成
        combined_img = Image.new("RGB", (combined_width, combined_height), "white")
        draw = ImageDraw.Draw(combined_img)

        # タイトルを描画
        try:
            # システムフォントを試す
            title_font = ImageFont.truetype("arial.ttf", 28)
            label_font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            try:
                title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
                label_font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except (OSError, IOError):
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()

        title = f"SDF Primitive Visualizations ({num_images} primitives)"

        # タイトルの位置を計算（中央揃え）
        bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = bbox[2] - bbox[0]
        title_x = (combined_width - title_width) // 2
        draw.text((title_x, 25), title, fill="black", font=title_font)

        # 各画像を配置
        for i, (img_path, primitive_name) in enumerate(
            zip(image_paths, valid_primitives)
        ):
            try:
                img = Image.open(img_path)

                # グリッド位置を計算
                row = i // grid_cols
                col = i % grid_cols

                # 配置位置を計算
                x = padding + col * (img_width + padding)
                y = title_height + padding + row * (img_height + padding)

                # 画像を貼り付け
                combined_img.paste(img, (x, y))
                img.close()

                # プリミティブ名を画像の下に描画
                text_bbox = draw.textbbox((0, 0), primitive_name, font=label_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x + (img_width - text_width) // 2
                text_y = y + img_height + 5

                # テキストの背景を白で塗りつぶし
                draw.rectangle(
                    [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + 20],
                    fill="white",
                )
                draw.text(
                    (text_x, text_y), primitive_name, fill="black", font=label_font
                )

            except Exception as e:
                print(f"    Warning: Could not process image {img_path}: {e}")
                continue

        # 結合画像を保存
        timestamp = os.path.basename(output_dir)
        if not timestamp or timestamp == "visualize_output":
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = os.path.join(combined_dir, f"combined_primitives_{timestamp}.png")
        combined_img.save(output_path, quality=95)
        print(f"  Combined primitive visualization saved: {output_path}")

        # 結果を辞書形式で返す
        results = {"2d_image": output_path}

        # 3D可視化の結合も実行
        if include_3d:
            print("  Also combining 3D visualizations...")

            # 3D isosurface結合
            isosurface_results = combine_3d_visualizations(
                valid_primitives, output_dir, "3d"
            )
            if isosurface_results:
                results["3d_isosurface_html"] = isosurface_results.get("html")
                if "png" in isosurface_results:
                    results["3d_isosurface_png"] = isosurface_results["png"]

            # 3D mesh結合
            mesh_results = combine_3d_visualizations(
                valid_primitives, output_dir, "mesh"
            )
            if mesh_results:
                results["3d_mesh_html"] = mesh_results.get("html")
                if "png" in mesh_results:
                    results["3d_mesh_png"] = mesh_results["png"]

        return results

    except ImportError:
        print("  Error: PIL (Pillow) not available, cannot combine primitive images")
        print("  Install with: pip install Pillow")
        return None
    except Exception as e:
        print(f"  Error combining primitive visualizations: {e}")
        return None


def combine_variation_images(image_paths, output_path, primitive_name, grid_cols=3):
    """
    複数のバリエーション画像を一枚に結合する

    Args:
        image_paths: 結合する画像ファイルパスのリスト
        output_path: 出力画像のパス
        primitive_name: プリミティブ名（タイトル用）
        grid_cols: グリッドの列数
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        if not image_paths:
            print(f"  Warning: No images to combine for {primitive_name}")
            return

        # 存在する画像ファイルのみをフィルタリング
        valid_paths = [path for path in image_paths if os.path.exists(path)]
        if not valid_paths:
            print(f"  Warning: No valid image files found for {primitive_name}")
            return

        # 最初の画像を読み込んでサイズを取得
        sample_img = Image.open(valid_paths[0])
        img_width, img_height = sample_img.size
        sample_img.close()

        # グリッドサイズを計算
        num_images = len(valid_paths)
        grid_rows = (num_images + grid_cols - 1) // grid_cols

        # タイトル用の余白
        title_height = 60
        padding = 10

        # 結合後の画像サイズ
        combined_width = grid_cols * img_width + (grid_cols + 1) * padding
        combined_height = (
            grid_rows * img_height + (grid_rows + 1) * padding + title_height
        )

        # 白い背景の新しい画像を作成
        combined_img = Image.new("RGB", (combined_width, combined_height), "white")
        draw = ImageDraw.Draw(combined_img)

        # タイトルを描画
        try:
            # システムフォントを試す
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except (OSError, IOError):
                font = ImageFont.load_default()

        title = f"{primitive_name} - Variations ({num_images} samples)"

        # タイトルの位置を計算（中央揃え）
        bbox = draw.textbbox((0, 0), title, font=font)
        title_width = bbox[2] - bbox[0]
        title_x = (combined_width - title_width) // 2
        draw.text((title_x, 20), title, fill="black", font=font)

        # 各画像を配置
        for i, img_path in enumerate(valid_paths):
            try:
                img = Image.open(img_path)

                # グリッド位置を計算
                row = i // grid_cols
                col = i % grid_cols

                # 配置位置を計算
                x = padding + col * (img_width + padding)
                y = title_height + padding + row * (img_height + padding)

                # 画像を貼り付け
                combined_img.paste(img, (x, y))
                img.close()

                # バリエーション番号を画像の上に描画
                try:
                    small_font = ImageFont.truetype("arial.ttf", 16)
                except (OSError, IOError):
                    try:
                        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
                    except (OSError, IOError):
                        small_font = ImageFont.load_default()

                variation_text = f"Var {i + 1}"
                text_bbox = draw.textbbox((0, 0), variation_text, font=small_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x + (img_width - text_width) // 2
                text_y = y - 25

                # テキストの背景を白で塗りつぶし
                draw.rectangle(
                    [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + 18],
                    fill="white",
                )
                draw.text(
                    (text_x, text_y), variation_text, fill="black", font=small_font
                )

            except Exception as e:
                print(f"    Warning: Could not process image {img_path}: {e}")
                continue

        # 結合画像を保存
        combined_img.save(output_path, quality=95)
        print(f"  Combined variation image saved: {output_path}")

    except ImportError:
        print("  Warning: PIL (Pillow) not available, cannot combine variation images")
        print("  Install with: pip install Pillow")
    except Exception as e:
        print(f"  Error combining variation images for {primitive_name}: {e}")


def generate_primitive_variations(
    output_dir="visualize_output",
    primitive_name="Sphere",
    num_variations=6,
    enable_3d=False,
):
    """
    指定されたプリミティブの複数のバリエーションを生成し、結合画像を作成

    Args:
        output_dir: 出力ディレクトリ
        primitive_name: 生成するプリミティブ名
        num_variations: バリエーション数
        enable_3d: 3D可視化を有効にするかどうか
    """

    # 出力ディレクトリを作成
    variations_dir = os.path.join(output_dir, "variations")
    os.makedirs(variations_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [64, 64, 64]
    device = torch.device("cpu")

    # 座標メッシュを作成
    zs = torch.linspace(0, grid_size[0] - 1, grid_size[0], dtype=torch.float32)
    ys = torch.linspace(0, grid_size[1] - 1, grid_size[1], dtype=torch.float32)
    xs = torch.linspace(0, grid_size[2] - 1, grid_size[2], dtype=torch.float32)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")

    # プリミティブクラスのマッピング（バリエーション分析対応）
    primitive_mapping = {
        # 基本プリミティブ
        "Sphere": (Sphere, {"center": (32.0, 32.0, 32.0)}),
        "Torus": (Torus, {"center": (32.0, 32.0, 32.0)}),
        "Cone": (Cone, {"center": (32.0, 32.0, 32.0)}),
        "Octahedron": (Octahedron, {"center": (32.0, 32.0, 32.0)}),
        "Cylinder": (Cylinder, {"center": (32.0, 32.0, 32.0)}),
        "ConvexCylinder": (ConvexCylinder, {"center": (32.0, 32.0, 32.0)}),
        "ConcaveCylinder": (ConcaveCylinder, {"center": (32.0, 32.0, 32.0)}),
        "ConeCylinder": (ConeCylinder, {"center": (32.0, 32.0, 32.0)}),
        # セクターポリゴンプリズム
        "TrianglePrism": (TrianglePrism, {"center": (32.0, 32.0, 32.0)}),
        "SquarePrism": (SquarePrism, {"center": (32.0, 32.0, 32.0)}),
        "PentagonPrism": (PentagonPrism, {"center": (32.0, 32.0, 32.0)}),
        "HexagonPrism": (HexagonPrism, {"center": (32.0, 32.0, 32.0)}),
        "HeptagonPrism": (HeptagonPrism, {"center": (32.0, 32.0, 32.0)}),
        "OctagonPrism": (OctagonPrism, {"center": (32.0, 32.0, 32.0)}),
        # コーンプリズム
        "TriangleConePrism": (TriangleConePrism, {"center": (32.0, 32.0, 32.0)}),
        "SquareConePrism": (SquareConePrism, {"center": (32.0, 32.0, 32.0)}),
        "PentagonConePrism": (PentagonConePrism, {"center": (32.0, 32.0, 32.0)}),
        "HexagonConePrism": (HexagonConePrism, {"center": (32.0, 32.0, 32.0)}),
        # 凸/凹プリズム
        "TriangleConvexPrism": (TriangleConvexPrism, {"center": (32.0, 32.0, 32.0)}),
        "SquareConvexPrism": (SquareConvexPrism, {"center": (32.0, 32.0, 32.0)}),
        "TriangleConcavePrism": (TriangleConcavePrism, {"center": (32.0, 32.0, 32.0)}),
        "SquareConcavePrism": (SquareConcavePrism, {"center": (32.0, 32.0, 32.0)}),
        # スタープリズム
        "FiveStarPrism": (FiveStarPrism, {"center": (32.0, 32.0, 32.0)}),
        "SixStarPrism": (SixStarPrism, {"center": (32.0, 32.0, 32.0)}),
        # トーラス系
        "SquareTorus": (SquareTorus, {"center": (32.0, 32.0, 32.0)}),
        "PentagonTorus": (PentagonTorus, {"center": (32.0, 32.0, 32.0)}),
        "HexagonTorus": (HexagonTorus, {"center": (32.0, 32.0, 32.0)}),
        "FiveStarTorus": (FiveStarTorus, {"center": (32.0, 32.0, 32.0)}),
        "SixStarTorus": (SixStarTorus, {"center": (32.0, 32.0, 32.0)}),
        # Revolution系
        "ThreeStarRevolution": (ThreeStarRevolution, {"center": (32.0, 32.0, 32.0)}),
        "FourStarRevolution": (FourStarRevolution, {"center": (32.0, 32.0, 32.0)}),
        "FiveStarRevolution": (FiveStarRevolution, {"center": (32.0, 32.0, 32.0)}),
        # Onioned系（代表的なもの）
        "OnionedCylinder": (OnionedCylinder, {"center": (32.0, 32.0, 32.0)}),
        "OnionedTrianglePrism": (OnionedTrianglePrism, {"center": (32.0, 32.0, 32.0)}),
        "OnionedSquarePrism": (OnionedSquarePrism, {"center": (32.0, 32.0, 32.0)}),
        "OnionedFiveStarPrism": (OnionedFiveStarPrism, {"center": (32.0, 32.0, 32.0)}),
        # Union系（代表的なもの）
        "SphereCylinderUnion": (SphereCylinderUnion, {"center": (32.0, 32.0, 32.0)}),
        "SphereTriangleUnion": (SphereTriangleUnion, {"center": (32.0, 32.0, 32.0)}),
        "FiveStarRevolutionCylinderUnion": (
            FiveStarRevolutionCylinderUnion,
            {"center": (32.0, 32.0, 32.0)},
        ),
        "FiveStarRevolutionPentagonUnion": (
            FiveStarRevolutionPentagonUnion,
            {"center": (32.0, 32.0, 32.0)},
        ),
    }

    if primitive_name not in primitive_mapping:
        print(
            f"Error: Primitive '{primitive_name}' not supported for variation analysis"
        )
        print(f"Supported primitives: {list(primitive_mapping.keys())}")
        return

    PrimClass, base_params = primitive_mapping[primitive_name]

    print(f"Generating {num_variations} variations of {primitive_name}...")

    variation_images = []
    variation_3d_mesh_images = []  # 3D mesh PNG画像専用のリスト

    for i in range(num_variations):
        print(f"  Processing variation {i + 1}/{num_variations}...")

        try:
            # プリミティブを生成（ランダムシードで異なるバリエーション）
            primitive = PrimClass(grid_size, device, **base_params)

            # SDFを計算
            sdf = primitive.sdf(X, Y, Z)

            # SDFを可視化用に変換
            sdf_vis = 128.0 / (torch.pow(torch.abs(sdf), 2.0) + 1.0)
            sdf_vis = torch.clamp(sdf_vis, 0.0, 128.0).to(torch.uint8)

            # セグメンテーションマスクを作成
            mask = (sdf <= 0).to(torch.uint8)

            # NumPy配列に変換
            sdf_np = sdf_vis.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # 個別の可視化を実行
            output_file = os.path.join(
                variations_dir, f"{primitive_name.lower()}_var_{i + 1:02d}.png"
            )
            visualize_sample((sdf_np, mask_np), output_file)

            variation_images.append(output_file)

            # 3D可視化も生成（enable_3dが有効な場合）
            if enable_3d:
                # 元のSDFデータをNumPy配列に変換（3D可視化用）
                sdf_original_np = sdf.cpu().numpy()

                # 3D isosurface visualization（HTMLのみ生成、結合画像には含めない）
                html_3d_file = os.path.join(
                    variations_dir, f"{primitive_name.lower()}_var_{i + 1:02d}_3d.html"
                )
                visualize_primitive_3d_isosurface(
                    sdf_original_np, html_3d_file, f"{primitive_name} Variation {i + 1}"
                )

                # 3D mesh visualization（mesh画像のみを結合画像用に収集）
                mesh_3d_file = os.path.join(
                    variations_dir,
                    f"{primitive_name.lower()}_var_{i + 1:02d}_mesh.html",
                )
                png_mesh_file = visualize_primitive_marching_cubes(
                    sdf_original_np, mesh_3d_file, f"{primitive_name} Variation {i + 1}"
                )
                if png_mesh_file:
                    variation_3d_mesh_images.append(png_mesh_file)

            # 統計情報を表示
            inside_count = (sdf < 0).sum().item()
            outside_count = (sdf > 0).sum().item()
            print(
                f"    Variation {i + 1}: inside={inside_count}, outside={outside_count}"
            )

        except Exception as e:
            print(f"    Error processing variation {i + 1}: {e}")
            continue

    # バリエーション画像を結合
    if variation_images:
        combined_output = os.path.join(
            variations_dir, f"{primitive_name.lower()}_variations_combined.png"
        )
        combine_variation_images(variation_images, combined_output, primitive_name)

        # 3D mesh画像を結合（3D mesh PNG画像が存在する場合）
        if enable_3d and variation_3d_mesh_images:
            combined_3d_output = os.path.join(
                variations_dir, f"{primitive_name.lower()}_variations_3d_combined.png"
            )
            combine_variation_images(
                variation_3d_mesh_images,
                combined_3d_output,
                f"{primitive_name} 3D Mesh",
            )

        print(f"\nVariation analysis completed for {primitive_name}!")
        print(
            f"Individual variations: {len(variation_images)} files in '{variations_dir}'"
        )
        print(f"Combined image: {combined_output}")
        if enable_3d:
            print("3D visualizations: HTML files generated for each variation")
            if variation_3d_mesh_images:
                print(f"Combined 3D mesh image: {combined_3d_output}")
            else:
                print(
                    "Note: 3D mesh PNG images not available (install kaleido for static 3D images)"
                )
    else:
        print(f"No variations were successfully generated for {primitive_name}")


def get_all_primitive_names():
    """
    システム内で利用可能な全プリミティブ名のリストを取得

    Returns:
        list: 全プリミティブ名のリスト
    """
    # generate_primitive_visualizations関数内で定義された全プリミティブを収集
    all_primitives = []

    # 基本プリミティブ
    basic_primitives = [
        "Sphere",
        "Torus",
        "Cone",
        "Octahedron",
        "Cylinder",
        "ConvexCylinder",
        "ConcaveCylinder",
        "ConeCylinder",
    ]

    # セクターポリゴンプリズム
    sector_polygon_primitives = [
        "SectorPolygonPrism",
        "TrianglePrism",
        "SquarePrism",
        "PentagonPrism",
        "HexagonPrism",
        "HeptagonPrism",
        "OctagonPrism",
        "NonagonPrism",
    ]

    # コーンプリズム
    cone_primitives = [
        "TriangleConePrism",
        "SquareConePrism",
        "PentagonConePrism",
        "HexagonConePrism",
    ]

    # 凸プリズム
    convex_primitives = ["TriangleConvexPrism", "SquareConvexPrism"]

    # 凹プリズム
    concave_primitives = ["TriangleConcavePrism", "SquareConcavePrism"]

    # スタープリズム
    star_primitives = ["FiveStarPrism", "SixStarPrism"]

    # トーラス系
    torus_primitives = [
        "SquareTorus",
        "PentagonTorus",
        "HexagonTorus",
        "HeptagonTorus",
        "OctagonTorus",
        "NonagonTorus",
        "FiveStarTorus",
        "SixStarTorus",
        "SevenStarTorus",
        "EightStarTorus",
    ]

    # Revolution系
    revolution_primitives = [
        "ThreeStarRevolution",
        "FourStarRevolution",
        "FiveStarRevolution",
    ]

    # Onioned Sector系
    onioned_sector_primitives = [
        "OnionedCylinder",
        "OnionedTrianglePrism",
        "OnionedSquarePrism",
        "OnionedPentagonPrism",
        "OnionedHexagonPrism",
        "OnionedTriangleConvexPrism",
        "OnionedSquareConvexPrism",
        "OnionedTriangleConcavePrism",
        "OnionedSquareConcavePrism",
        "OnionedTriangleConePrism",
        "OnionedSquareConePrism",
    ]

    # Onioned Star系
    onioned_star_primitives = [
        "OnionedFiveStarPrism",
        "OnionedSixStarPrism",
        "OnionedSevenStarPrism",
        "OnionedEightStarPrism",
        "OnionedFiveStarConvexPrism",
        "OnionedSixStarConvexPrism",
        "OnionedSevenStarConvexPrism",
        "OnionedEightStarConvexPrism",
        "OnionedFiveStarConcavePrism",
        "OnionedSixStarConcavePrism",
        "OnionedSevenStarConcavePrism",
        "OnionedEightStarConcavePrism",
        "OnionedFiveStarConePrism",
        "OnionedSixStarConePrism",
        "OnionedSevenStarConePrism",
        "OnionedEightStarConePrism",
    ]

    # Union系
    union_primitives = [
        "SphereTriangleUnion",
        "SphereSquareUnion",
        "SpherePentagonUnion",
        "SphereCylinderUnion",
        "ThreeStarRevolutionTriangleUnion",
        "ThreeStarRevolutionSquareUnion",
        "ThreeStarRevolutionPentagonUnion",
        "ThreeStarRevolutionCylinderUnion",
        "FourStarRevolutionTriangleUnion",
        "FourStarRevolutionSquareUnion",
        "FourStarRevolutionPentagonUnion",
        "FourStarRevolutionCylinderUnion",
        "FiveStarRevolutionTriangleUnion",
        "FiveStarRevolutionSquareUnion",
        "FiveStarRevolutionPentagonUnion",
        "FiveStarRevolutionCylinderUnion",
    ]

    # 全プリミティブを結合
    all_primitives.extend(basic_primitives)
    all_primitives.extend(sector_polygon_primitives)
    all_primitives.extend(cone_primitives)
    all_primitives.extend(convex_primitives)
    all_primitives.extend(concave_primitives)
    all_primitives.extend(star_primitives)
    all_primitives.extend(torus_primitives)
    all_primitives.extend(revolution_primitives)
    all_primitives.extend(onioned_sector_primitives)
    all_primitives.extend(onioned_star_primitives)
    all_primitives.extend(union_primitives)

    return sorted(list(set(all_primitives)))  # 重複を除去してソート


def generate_primitive_visualizations(
    output_dir="visualize_output",
    primitive_type="all",
    enable_3d=True,
    auto_combine=False,
    grid_cols=3,
    combine_3d=False,
):
    """各プリミティブを個別に生成し、可視化結果を保存する

    Args:
        output_dir: 出力ディレクトリ
        primitive_type: 生成するプリミティブのタイプ ("all", "star", "basic", "polygon")
        enable_3d: 3D可視化を有効にするかどうか
    """

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [64, 64, 64]  # より細かなグリッドで詳細な可視化
    device = torch.device("cpu")

    # 座標メッシュを作成
    zs = torch.linspace(0, grid_size[0] - 1, grid_size[0], dtype=torch.float32)
    ys = torch.linspace(0, grid_size[1] - 1, grid_size[1], dtype=torch.float32)
    xs = torch.linspace(0, grid_size[2] - 1, grid_size[2], dtype=torch.float32)
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")

    # 基本プリミティブ
    basic_primitives = [
        ("Sphere", Sphere, {"center": (32.0, 32.0, 32.0)}),
        (
            "Torus",
            Torus,
            {"center": (32.0, 32.0, 32.0)},
        ),
        (
            "Cone",
            Cone,
            {"center": (32.0, 32.0, 32.0)},
        ),
        (
            "Octahedron",
            Octahedron,
            {"center": (32.0, 32.0, 32.0)},
        ),
        (
            "Cylinder",
            Cylinder,
            {"center": (32.0, 32.0, 32.0)},
        ),
        (
            "ConvexCylinder",
            ConvexCylinder,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "ConcaveCylinder",
            ConcaveCylinder,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "ConeCylinder",
            ConeCylinder,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # セクターポリゴンプリズム
    sector_polygon_primitives = [
        (
            "SectorPolygonPrism",
            SectorPolygonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "TrianglePrism",
            TrianglePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SquarePrism",
            SquarePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "PentagonPrism",
            PentagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "HexagonPrism",
            HexagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "HeptagonPrism",
            HeptagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OctagonPrism",
            OctagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "NonagonPrism",
            NonagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # コーンプリズム
    cone_primitives = [
        (
            "TriangleConePrism",
            TriangleConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SquareConePrism",
            SquareConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "PentagonConePrism",
            PentagonConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "HexagonConePrism",
            HexagonConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # 凸プリズム
    convex_primitives = [
        (
            "TriangleConvexPrism",
            TriangleConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SquareConvexPrism",
            SquareConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # 凹プリズム
    concave_primitives = [
        (
            "TriangleConcavePrism",
            TriangleConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SquareConcavePrism",
            SquareConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # スタープリズム
    star_primitives = [
        (
            "FiveStarPrism",
            FiveStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SixStarPrism",
            SixStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # トーラスプリミティブ
    torus_primitives = [
        # セクターポリゴントーラス
        (
            "SquareTorus",
            SquareTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "PentagonTorus",
            PentagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "HexagonTorus",
            HexagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "HeptagonTorus",
            HeptagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OctagonTorus",
            OctagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "NonagonTorus",
            NonagonTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # スタートーラス
        (
            "FiveStarTorus",
            FiveStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SixStarTorus",
            SixStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SevenStarTorus",
            SevenStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "EightStarTorus",
            EightStarTorus,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # Revolution（回転体）プリミティブ
    revolution_primitives = [
        (
            "ThreeStarRevolution",
            ThreeStarRevolution,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FourStarRevolution",
            FourStarRevolution,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FiveStarRevolution",
            FiveStarRevolution,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # Onioned Sector Polygonプリミティブ（オニオン形状）
    onioned_sector_primitives = [
        # 基本Onionedプリズム
        (
            "OnionedCylinder",
            OnionedCylinder,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedTrianglePrism",
            OnionedTrianglePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSquarePrism",
            OnionedSquarePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedPentagonPrism",
            OnionedPentagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedHexagonPrism",
            OnionedHexagonPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # Onioned凸プリズム
        (
            "OnionedTriangleConvexPrism",
            OnionedTriangleConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSquareConvexPrism",
            OnionedSquareConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # Onioned凹プリズム
        (
            "OnionedTriangleConcavePrism",
            OnionedTriangleConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSquareConcavePrism",
            OnionedSquareConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # Onionedコーンプリズム
        (
            "OnionedTriangleConePrism",
            OnionedTriangleConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSquareConePrism",
            OnionedSquareConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # Onioned Star Primitivesプリミティブ（オニオン形状の星形）
    onioned_star_primitives = [
        # OnionedStarプリズム（基本）
        (
            "OnionedFiveStarPrism",
            OnionedFiveStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSixStarPrism",
            OnionedSixStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSevenStarPrism",
            OnionedSevenStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedEightStarPrism",
            OnionedEightStarPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # OnionedStar凸プリズム
        (
            "OnionedFiveStarConvexPrism",
            OnionedFiveStarConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSixStarConvexPrism",
            OnionedSixStarConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSevenStarConvexPrism",
            OnionedSevenStarConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedEightStarConvexPrism",
            OnionedEightStarConvexPrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # OnionedStar凹プリズム
        (
            "OnionedFiveStarConcavePrism",
            OnionedFiveStarConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSixStarConcavePrism",
            OnionedSixStarConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSevenStarConcavePrism",
            OnionedSevenStarConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedEightStarConcavePrism",
            OnionedEightStarConcavePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # OnionedStarコーンプリズム
        (
            "OnionedFiveStarConePrism",
            OnionedFiveStarConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSixStarConePrism",
            OnionedSixStarConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedSevenStarConePrism",
            OnionedSevenStarConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "OnionedEightStarConePrism",
            OnionedEightStarConePrism,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # Union Primitivesプリミティブ（組み合わせ形状）
    union_primitives = [
        # Sphere based unions
        (
            "SphereTriangleUnion",
            SphereTriangleUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SphereSquareUnion",
            SphereSquareUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SpherePentagonUnion",
            SpherePentagonUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "SphereCylinderUnion",
            SphereCylinderUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # ThreeStarRevolution based unions
        (
            "ThreeStarRevolutionTriangleUnion",
            ThreeStarRevolutionTriangleUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "ThreeStarRevolutionSquareUnion",
            ThreeStarRevolutionSquareUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "ThreeStarRevolutionPentagonUnion",
            ThreeStarRevolutionPentagonUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "ThreeStarRevolutionCylinderUnion",
            ThreeStarRevolutionCylinderUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # FourStarRevolution based unions
        (
            "FourStarRevolutionTriangleUnion",
            FourStarRevolutionTriangleUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FourStarRevolutionSquareUnion",
            FourStarRevolutionSquareUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FourStarRevolutionPentagonUnion",
            FourStarRevolutionPentagonUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FourStarRevolutionCylinderUnion",
            FourStarRevolutionCylinderUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        # FiveStarRevolution based unions
        (
            "FiveStarRevolutionTriangleUnion",
            FiveStarRevolutionTriangleUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FiveStarRevolutionSquareUnion",
            FiveStarRevolutionSquareUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FiveStarRevolutionPentagonUnion",
            FiveStarRevolutionPentagonUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
        (
            "FiveStarRevolutionCylinderUnion",
            FiveStarRevolutionCylinderUnion,
            {
                "center": (32.0, 32.0, 32.0),
            },
        ),
    ]

    # プリミティブタイプに応じて選択
    if primitive_type == "star":
        selected_primitives = star_primitives
        print(
            f"Generating {len(selected_primitives)} star primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "basic":
        selected_primitives = basic_primitives
        print(
            f"Generating {len(selected_primitives)} basic primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "polygon":
        selected_primitives = (
            sector_polygon_primitives
            + cone_primitives
            + convex_primitives
            + concave_primitives
        )
        print(
            f"Generating {len(selected_primitives)} polygon primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "torus":
        selected_primitives = torus_primitives
        print(
            f"Generating {len(selected_primitives)} torus primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "revolution":
        selected_primitives = revolution_primitives
        print(
            f"Generating {len(selected_primitives)} revolution primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "onioned_sector":
        selected_primitives = onioned_sector_primitives
        print(
            f"Generating {len(selected_primitives)} onioned sector primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "onioned_star":
        selected_primitives = onioned_star_primitives
        print(
            f"Generating {len(selected_primitives)} onioned star primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "onioned":
        selected_primitives = onioned_sector_primitives + onioned_star_primitives
        print(
            f"Generating {len(selected_primitives)} onioned primitive visualizations in '{output_dir}'..."
        )
    elif primitive_type == "union":
        selected_primitives = union_primitives
        print(
            f"Generating {len(selected_primitives)} union primitive visualizations in '{output_dir}'..."
        )
    else:  # "all" or default
        selected_primitives = (
            basic_primitives
            + sector_polygon_primitives
            + cone_primitives
            + convex_primitives
            + concave_primitives
            + star_primitives
            + torus_primitives
            + revolution_primitives
            + onioned_sector_primitives
            + onioned_star_primitives
            + union_primitives
        )
        print(
            f"Generating {len(selected_primitives)} primitive visualizations in '{output_dir}'..."
        )

    for name, PrimClass, params in selected_primitives:
        print(f"Processing {name}...")

        try:
            # プリミティブを生成（シード値は固定しない）
            primitive = PrimClass(grid_size, device, **params)

            # SDFを計算
            sdf = primitive.sdf(X, Y, Z)

            # SDFを可視化用に変換（0-128の範囲）
            sdf_vis = 128.0 / (torch.pow(torch.abs(sdf), 2.0) + 1.0)
            sdf_vis = torch.clamp(sdf_vis, 0.0, 128.0).to(torch.uint8)

            # セグメンテーションマスクを作成（オブジェクトIDは1）
            mask = (sdf <= 0).to(torch.uint8)

            # NumPy配列に変換
            sdf_np = sdf_vis.cpu().numpy()
            mask_np = mask.cpu().numpy()

            # 従来の可視化を実行
            output_file = os.path.join(output_dir, f"{name.lower()}_visualization.png")
            visualize_sample((sdf_np, mask_np), output_file)

            # 3D可視化を条件付きで実行
            if enable_3d:
                # 3D isosurface可視化を追加
                visualize_primitive_3d_isosurface(sdf, output_file, name)

                # Marching Cubesメッシュ可視化も試行
                visualize_primitive_marching_cubes(sdf, output_file, name)

            # 統計情報を表示
            inside_count = (sdf < 0).sum().item()
            outside_count = (sdf > 0).sum().item()

            print(f"  {name}: inside={inside_count}, outside={outside_count}")
            print(f"  Saved: {output_file}")
            print(f"  Slice: {output_file.replace('.png', '_slice.png')}")

        except Exception as e:
            print(f"  Error processing {name}: {e}")
            continue

    print(f"\nAll visualizations saved in '{output_dir}' directory!")

    # 自動結合オプションが有効な場合
    if auto_combine and selected_primitives:
        print(
            f"\nAuto-combining {len(selected_primitives)} primitive visualizations..."
        )
        primitive_names = [name for name, _, _ in selected_primitives]
        results = combine_primitive_visualizations(
            primitive_names, output_dir, grid_cols, combine_3d
        )
        if results:
            if isinstance(results, dict):
                for viz_type, path in results.items():
                    print(
                        f"Combined {viz_type} visualization automatically saved: {path}"
                    )
            else:
                print(f"Combined visualization automatically saved: {results}")
        else:
            print(
                "Auto-combine failed. Some visualizations may not have been generated successfully."
            )


def generate_dataset_samples(output_dir="visualize_output", num_samples=5):
    """データセットサンプルの可視化を生成"""
    from src.fdslxsdf4seg.generate_sdf_dataset import SDFSegmentationDataset

    # 出力ディレクトリを作成
    samples_dir = os.path.join(output_dir, "dataset_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # グリッドサイズとデバイス設定
    grid_size = [64, 64, 64]
    device = torch.device("cpu")

    # データセットを作成
    ds = SDFSegmentationDataset(
        grid_size=grid_size,
        num_volumes=num_samples,
        min_objects=20,
        max_objects=20,
        device=device,
    )

    print(f"\nGenerating {num_samples} dataset samples in '{samples_dir}'...")

    for i in range(num_samples):
        print(f"Processing sample {i + 1}/{num_samples}...")

        x, y = ds[i]

        # 可視化を実行
        output_file = os.path.join(samples_dir, f"dataset_sample_{i:03d}.png")
        visualize_sample((x, y), output_file)

        # オブジェクト情報を表示
        unique_objects = np.unique(y)
        print(f"  Sample {i}: objects={unique_objects}")
        print(f"  Saved: {output_file}")
        print(f"  Slice: {output_file.replace('.png', '_slice.png')}")

    print(f"\nAll dataset samples saved in '{samples_dir}' directory!")


def print_combine_usage_examples():
    """結合機能の使用例を表示"""
    print("\n=== Primitive Combination Examples ===")
    print("1. Generate and auto-combine basic primitives (2D only):")
    print(
        "   python visualize_primitives.py --primitives --primitive_type basic --auto_combine"
    )
    print("\n2. Generate and auto-combine with 3D visualizations:")
    print(
        "   python visualize_primitives.py --primitives --primitive_type basic --auto_combine --combine_3d --3d"
    )
    print("\n3. Combine specific primitives (must be generated first):")
    print(
        "   python visualize_primitives.py --combine --combine_primitives Sphere Torus Cone"
    )
    print("\n4. Combine with 3D visualizations (HTML + PNG):")
    print(
        "   python visualize_primitives.py --combine --combine_primitives Sphere Torus Cone --combine_3d"
    )
    print("\n5. Generate all primitives and combine with custom grid:")
    print("   python visualize_primitives.py --primitives --auto_combine --grid_cols 4")
    print("\n6. Combine star primitives with 3D (HTML + PNG):")
    print(
        "   python visualize_primitives.py --combine --combine_primitives FiveStarPrism SixStarPrism FiveStarTorus --combine_3d"
    )
    print("\n7. Combine ALL available primitives (may take time):")
    print(
        "   python visualize_primitives.py --combine --combine_all_primitives --grid_cols 6"
    )
    print("\n8. List all available primitive names:")
    print("   python visualize_primitives.py --list_all_primitives")
    print("\n=== 3D Combination Features ===")
    print("When using --combine_3d, the following files are generated:")
    print("• 2D combined image: PNG grid of all selected primitives")
    print("• 3D HTML files: Interactive combined visualizations in browser")
    print("• 3D PNG files: Static combined images of 3D visualizations")
    print("  - combined_3d_png_TIMESTAMP.png (isosurface)")
    print("  - combined_mesh_png_TIMESTAMP.png (mesh)")
    print("\n=== Available Primitive Types ===")
    print(
        "Basic: Sphere, Torus, Cone, Octahedron, Cylinder, ConvexCylinder, ConcaveCylinder, ConeCylinder"
    )
    print(
        "Polygon: TrianglePrism, SquarePrism, PentagonPrism, HexagonPrism, HeptagonPrism, OctagonPrism"
    )
    print("Star: FiveStarPrism, SixStarPrism, FiveStarTorus, SixStarTorus")
    print(
        f"Total: {len(get_all_primitive_names())} primitives available (use --list_all_primitives to see all)"
    )
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate primitive and dataset visualizations with combination features"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualize_output",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--primitives",
        action="store_true",
        help="Generate individual primitive visualizations",
    )
    parser.add_argument(
        "--primitive_type",
        type=str,
        choices=[
            "all",
            "star",
            "basic",
            "polygon",
            "torus",
            "revolution",
            "onioned",
            "onioned_sector",
            "onioned_star",
            "union",
        ],
        default="all",
        help="Type of primitives to generate (all/star/basic/polygon/torus/revolution/onioned/onioned_sector/onioned_star/union)",
    )
    parser.add_argument(
        "--dataset", action="store_true", help="Generate dataset sample visualizations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of dataset samples to generate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate both primitives and dataset samples",
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        help="Generate 3D isosurface visualizations (enabled by default)",
    )
    parser.add_argument(
        "--variations",
        action="store_true",
        help="Generate variations of a specific primitive",
    )
    parser.add_argument(
        "--variation_primitive",
        type=str,
        default="Sphere",
        help="Primitive name for variation analysis (e.g., Sphere, FiveStarPrism, etc.)",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=6,
        help="Number of variations to generate (default: 6)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine multiple primitive visualizations into a single image",
    )
    parser.add_argument(
        "--combine_primitives",
        nargs="+",
        help="List of primitive names to combine (e.g., Sphere Torus Cone)",
    )
    parser.add_argument(
        "--combine_all_primitives",
        action="store_true",
        help="Combine ALL available primitives into a single image (may take time)",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=3,
        help="Number of columns in the combined grid (default: 3)",
    )
    parser.add_argument(
        "--auto_combine",
        action="store_true",
        help="Automatically combine generated primitives into a single image",
    )
    parser.add_argument(
        "--combine_3d",
        action="store_true",
        help="Also combine 3D visualizations (HTML files + PNG images) when combining",
    )
    parser.add_argument(
        "--help_combine",
        action="store_true",
        help="Show detailed usage examples for primitive combination features",
    )
    parser.add_argument(
        "--list_all_primitives",
        action="store_true",
        help="List all available primitive names and exit",
    )

    args = parser.parse_args()

    if args.help_combine:
        print_combine_usage_examples()
        exit(0)

    if args.list_all_primitives:
        all_primitives = get_all_primitive_names()
        print(f"\n=== All Available Primitives ({len(all_primitives)} total) ===")

        # カテゴリ別に整理して表示
        categories = {
            "Basic": [
                p
                for p in all_primitives
                if any(
                    basic in p
                    for basic in ["Sphere", "Torus", "Cone", "Octahedron", "Cylinder"]
                )
            ],
            "Polygon Prisms": [
                p
                for p in all_primitives
                if "Prism" in p
                and not any(
                    x in p for x in ["Star", "Onioned", "Convex", "Concave", "Cone"]
                )
            ],
            "Cone Prisms": [
                p for p in all_primitives if "ConePrism" in p and "Onioned" not in p
            ],
            "Convex Prisms": [
                p for p in all_primitives if "ConvexPrism" in p and "Onioned" not in p
            ],
            "Concave Prisms": [
                p for p in all_primitives if "ConcavePrism" in p and "Onioned" not in p
            ],
            "Star Prisms": [
                p for p in all_primitives if "StarPrism" in p and "Onioned" not in p
            ],
            "Torus Shapes": [p for p in all_primitives if "Torus" in p],
            "Revolution Shapes": [
                p for p in all_primitives if "Revolution" in p and "Union" not in p
            ],
            "Onioned Shapes": [p for p in all_primitives if "Onioned" in p],
            "Union Shapes": [p for p in all_primitives if "Union" in p],
        }

        for category, primitives in categories.items():
            if primitives:
                print(f"\n{category} ({len(primitives)} items):")
                for i, primitive in enumerate(sorted(primitives), 1):
                    print(f"  {i:2d}. {primitive}")

        print("\n💡 Usage Examples:")
        print("   # Combine specific primitives:")
        print(
            "   python visualize_primitives.py --combine --combine_primitives Sphere Torus Cone"
        )
        print("   # Combine ALL primitives (may take time):")
        print(
            "   python visualize_primitives.py --combine --combine_all_primitives --grid_cols 8"
        )
        print()
        exit(0)

    if args.all:
        args.primitives = True
        args.dataset = True

    # combine_all_primitivesが指定された場合はcombineも有効にする
    if args.combine_all_primitives:
        args.combine = True

    if not (args.primitives or args.dataset or args.variations or args.combine):
        print(
            "Please specify --primitives, --dataset, --variations, --combine, --combine_all_primitives, or --all"
        )
        parser.print_help()
        exit(1)

    if args.primitives:
        # 3D可視化はデフォルトで有効（明示的に無効にしない限り）
        enable_3d = getattr(args, "3d", True)
        generate_primitive_visualizations(
            args.output_dir,
            args.primitive_type,
            enable_3d,
            args.auto_combine,
            args.grid_cols,
            args.combine_3d,
        )

    if args.dataset:
        generate_dataset_samples(args.output_dir, args.num_samples)

    if args.variations:
        # 正しい属性名で3Dフラグを取得
        enable_3d = getattr(args, "3d", False)
        generate_primitive_variations(
            args.output_dir, args.variation_primitive, args.num_variations, enable_3d
        )

    if args.combine:
        if args.combine_all_primitives:
            # 全プリミティブを結合
            all_primitives = get_all_primitive_names()
            print(f"Combining ALL {len(all_primitives)} available primitives...")
            print("⚠️  This may take some time and require significant memory!")
            print(
                "💡 Consider using a larger grid_cols value (default: 3) for better layout"
            )

            # 確認メッセージ
            try:
                import time

                print("Starting in 3 seconds... (Press Ctrl+C to cancel)")
                time.sleep(3)
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                exit(0)

            results = combine_primitive_visualizations(
                all_primitives,
                args.output_dir,
                args.grid_cols,
                args.combine_3d,
            )
            if results:
                if isinstance(results, dict):
                    for viz_type, path in results.items():
                        print(f"Combined {viz_type} visualization saved: {path}")
                else:
                    print(f"Combined visualization saved: {results}")
                print(f"✅ Successfully combined all {len(all_primitives)} primitives!")
            else:
                print(
                    "❌ Failed to combine primitives. Some visualizations may not exist."
                )
                print("💡 Generate them first with: --primitives --primitive_type all")
        elif args.combine_primitives:
            # 指定されたプリミティブを結合
            print(f"Combining {len(args.combine_primitives)} specified primitives...")
            results = combine_primitive_visualizations(
                args.combine_primitives,
                args.output_dir,
                args.grid_cols,
                args.combine_3d,
            )
            if results:
                if isinstance(results, dict):
                    for viz_type, path in results.items():
                        print(f"Combined {viz_type} visualization saved: {path}")
                else:
                    print(f"Combined visualization saved: {results}")
        else:
            # デフォルトで代表的なプリミティブを結合
            default_primitives = [
                "Sphere",
                "Torus",
                "Cone",
                "Octahedron",
                "Cylinder",
                "TrianglePrism",
                "SquarePrism",
                "PentagonPrism",
                "HexagonPrism",
                "FiveStarPrism",
                "SixStarPrism",
            ]
            print(f"Combining {len(default_primitives)} default primitives...")
            results = combine_primitive_visualizations(
                default_primitives, args.output_dir, args.grid_cols, args.combine_3d
            )
            if results:
                if isinstance(results, dict):
                    for viz_type, path in results.items():
                        print(f"Combined {viz_type} visualization saved: {path}")
                else:
                    print(f"Combined visualization saved: {results}")
            else:
                print(
                    "Some primitive visualizations may not exist. Generate them first with --primitives."
                )
