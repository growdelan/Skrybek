"""Install Skrybek in ~/Applications and register it with Launch Services."""
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory

import AppKit as kit


def install_icon(resources):
    """Draw the same microphone glyph used in the recording control."""
    with TemporaryDirectory() as directory:
        iconset = Path(directory) / "Skrybek.iconset"
        iconset.mkdir()
        for points in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                size = points * scale
                image = kit.NSImage.alloc().initWithSize_(kit.NSMakeSize(size, size))
                image.lockFocus()
                inset = size * 0.09
                rect = kit.NSMakeRect(inset, inset, size - 2 * inset, size - 2 * inset)
                background = kit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, size * 0.19, size * 0.19
                )
                gradient = kit.NSGradient.alloc().initWithStartingColor_endingColor_(
                    kit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.13, 0.66, 0.62, 1),
                    kit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.04, 0.28, 0.32, 1),
                )
                gradient.drawInBezierPath_angle_(background, -60)
                symbol = kit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("mic.fill", None)
                config = kit.NSImageSymbolConfiguration.configurationWithPaletteColors_([kit.NSColor.whiteColor()])
                symbol = symbol.imageWithSymbolConfiguration_(config)
                width, height = symbol.size()
                ratio = size * 0.48 / height
                frame = kit.NSMakeRect((size - width * ratio) / 2, size * 0.26, width * ratio, size * 0.48)
                symbol.drawInRect_(frame)
                image.unlockFocus()
                bitmap = kit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                png = bitmap.representationUsingType_properties_(kit.NSBitmapImageFileTypePNG, {})
                suffix = "@2x" if scale == 2 else ""
                png.writeToFile_atomically_(str(iconset / f"icon_{points}x{points}{suffix}.png"), True)
        subprocess.run(["/usr/bin/iconutil", "-c", "icns", str(iconset),
                        "-o", str(resources / "Skrybek.icns")], check=True)


def install():
    root = Path(__file__).resolve().parents[1]
    ffmpeg = shutil.which("ffmpeg")
    path = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    if ffmpeg:
        path.insert(0, str(Path(ffmpeg).parent))
    path.insert(1, "/opt/homebrew/opt/ffmpeg-full/bin")
    app = Path.home() / "Applications" / "Skrybek.app"
    contents = app / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(exist_ok=True)
    install_icon(resources)
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump({
            "CFBundleName": "Skrybek",
            "CFBundleDisplayName": "Skrybek",
            "CFBundleIdentifier": "pl.gohan.skrybek",
            "CFBundleVersion": "1",
            "CFBundleShortVersionString": "0.4.0",
            "CFBundlePackageType": "APPL",
            "CFBundleExecutable": "launcher",
            "CFBundleIconFile": "Skrybek.icns",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": "Skrybek nagrywa mikrofon, aby lokalnie rozpoznawać mowę.",
        }, handle)
    launcher = macos / "launcher"
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    if not library.is_file():
        raise SystemExit(f"Nie znaleziono biblioteki Pythona: {library}")
    definitions = {
        "PROJECT_DIR": str(root),
        "PROJECT_PATH": os.pathsep.join(path),
        "PYTHON_EXECUTABLE": sys.executable,
        "PYTHON_LIBRARY": str(library),
    }
    with TemporaryDirectory() as directory:
        executable = Path(directory) / "launcher"
        subprocess.run([
            "/usr/bin/xcrun", "clang", "-Wall", "-Wextra", "-o", str(executable),
            *[f"-D{name}={json.dumps(value, ensure_ascii=False)}" for name, value in definitions.items()],
            str(Path(__file__).with_name("launcher.c")),
        ], check=True)
        os.replace(executable, launcher)
    subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(app)], check=True)
    subprocess.run([
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister",
        "-f", str(app),
    ], check=True)
    subprocess.run(["/usr/bin/mdimport", str(app)], check=True)
    print(f"Zainstalowano Skrybka w {app}")


if __name__ == "__main__":
    install()
