"""Native macOS window for recording, correcting and editing dictation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import AppKit as kit
from objc import python_method
from AVFoundation import (
    AVAuthorizationStatusAuthorized,
    AVAuthorizationStatusNotDetermined,
    AVCaptureDevice,
    AVMediaTypeAudio,
)
from Foundation import NSBundle, NSObject, NSMakeRange, NSUserDefaults

from main import (
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_WHISPER_MODEL,
    LiteRTError,
    copy_to_clipboard,
    process_text_with_litert_subprocess,
    record_until_silence,
    resolve_litert_system_prompt,
    save_wav_pcm16,
    transcribe_audio,
)

BUNDLE_ID = "pl.gohan.skrybek"
PREFERENCES = (
    ("whisper", "Model Whisper", DEFAULT_WHISPER_MODEL),
    ("parakeet", "Model Parakeet", DEFAULT_PARAKEET_MODEL),
    ("model", "Plik modelu LiteRT", ""),
    ("prompt", "Prompt lub ścieżka do pliku", ""),
)


def label(text, size=13, weight=kit.NSFontWeightRegular, secondary=False):
    view = kit.NSTextField.labelWithString_(text)
    view.setFont_(kit.NSFont.systemFontOfSize_weight_(size, weight))
    if secondary:
        view.setTextColor_(kit.NSColor.secondaryLabelColor())
    return view


def stack(views, *, vertical=False, spacing=10):
    view = kit.NSStackView.stackViewWithViews_(views)
    view.setOrientation_(kit.NSUserInterfaceLayoutOrientationVertical if vertical
                         else kit.NSUserInterfaceLayoutOrientationHorizontal)
    view.setAlignment_(kit.NSLayoutAttributeLeading if vertical else kit.NSLayoutAttributeCenterY)
    view.setSpacing_(spacing)
    return view


def button(title, target, action, symbol=None):
    view = kit.NSButton.buttonWithTitle_target_action_(title, target, action)
    view.setControlSize_(kit.NSControlSizeLarge)
    if symbol:
        view.setImage_(kit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None))
        view.setImagePosition_(kit.NSImageLeading)
    return view


class Desktop(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        if NSBundle.mainBundle().bundleIdentifier() == BUNDLE_ID:
            self.prefs = NSUserDefaults.standardUserDefaults()
        else:
            self.prefs = NSUserDefaults.alloc().initWithSuiteName_(BUNDLE_ID)
        self.prefs.registerDefaults_({key: default for key, _, default in PREFERENCES})
        self.state = "idle"
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.stop = Event()

        style = (kit.NSWindowStyleMaskTitled | kit.NSWindowStyleMaskClosable |
                 kit.NSWindowStyleMaskMiniaturizable | kit.NSWindowStyleMaskResizable |
                 kit.NSWindowStyleMaskFullSizeContentView)
        self.window = kit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            kit.NSMakeRect(0, 0, 820, 610), style, kit.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Skrybek")
        self.window.setTitleVisibility_(kit.NSWindowTitleHidden)
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setMinSize_(kit.NSMakeSize(650, 480))
        self.window.setDelegate_(self)
        background = kit.NSBox.alloc().initWithFrame_(self.window.contentView().bounds())
        background.setBoxType_(kit.NSBoxCustom)
        background.setBorderWidth_(0)
        background.setFillColor_(kit.NSColor.windowBackgroundColor())
        self.window.setContentView_(background)

        self.backend = kit.NSPopUpButton.alloc().initWithFrame_pullsDown_(kit.NSZeroRect, False)
        self.backend.addItemsWithTitles_(["Whisper", "Parakeet"])
        self.backend.selectItemWithTitle_(self.prefs.stringForKey_("backend") or "Whisper")
        self.backend.setAccessibilityLabel_("Model rozpoznawania mowy")
        self.backend.setTarget_(self)
        self.backend.setAction_("optionsChanged:")
        self.llm = kit.NSButton.checkboxWithTitle_target_action_("Korekta LiteRT", self, "optionsChanged:")
        self.llm.setState_(kit.NSControlStateValueOn if self.prefs.boolForKey_("correction") else
                           kit.NSControlStateValueOff)
        self.settings = button("Ustawienia", self, "showSettings:", "gearshape")

        self.editor = kit.NSTextView.scrollableTextView().documentView()
        self.editor.setRichText_(False)
        self.editor.setAllowsUndo_(True)
        self.editor.setAutomaticQuoteSubstitutionEnabled_(False)
        self.editor.setAutomaticDashSubstitutionEnabled_(False)
        self.editor.setFont_(kit.NSFont.systemFontOfSize_(16))
        self.editor.setTextContainerInset_(kit.NSMakeSize(18, 15))
        self.editor.setAccessibilityLabel_("Tekst transkrypcji")
        self.editor.setDelegate_(self)
        scroll = self.editor.enclosingScrollView()
        scroll.setBorderType_(kit.NSNoBorder)
        editor_frame = kit.NSBox.alloc().init()
        editor_frame.setBoxType_(kit.NSBoxCustom)
        editor_frame.setBorderColor_(kit.NSColor.separatorColor())
        editor_frame.setBorderWidth_(1)
        editor_frame.setCornerRadius_(12)
        editor_frame.setFillColor_(kit.NSColor.textBackgroundColor())
        editor_frame.setContentViewMargins_(kit.NSMakeSize(8, 5))
        editor_frame.setContentView_(scroll)

        self.record = button("Rozpocznij nagranie", self, "recordAction:", "mic.fill")
        self.record.setKeyEquivalent_("\r")
        self.record.setKeyEquivalentModifierMask_(kit.NSEventModifierFlagCommand)
        self.correct = button("Popraw tekst", self, "correctAction:", "wand.and.stars")
        self.copy = button("Kopiuj tekst", self, "copyAction:", "doc.on.doc")
        self.count = label("0 znaków", 12, secondary=True)
        self.status = label("Gotowy do nagrania", 12, secondary=True)
        self.status.setLineBreakMode_(kit.NSLineBreakByTruncatingTail)
        self.status.setToolTip_("Gotowy do nagrania")
        self.spinner = kit.NSProgressIndicator.alloc().init()
        self.spinner.setStyle_(kit.NSProgressIndicatorStyleSpinning)
        self.spinner.setControlSize_(kit.NSControlSizeSmall)
        self.spinner.setDisplayedWhenStopped_(False)

        title = label("Skrybek", 35, kit.NSFontWeightBold)
        heading = stack([title, label("Dyktowanie i edycja na Macu", 14, secondary=True)],
                        vertical=True, spacing=4)
        header = stack([heading, kit.NSView.alloc().init(), self.settings])
        options = stack([label("Rozpoznawanie mowy", weight=kit.NSFontWeightMedium), self.backend, self.llm],
                        spacing=14)
        editor_header = stack([label("Transkrypcja", 15, kit.NSFontWeightSemibold),
                               kit.NSView.alloc().init(), self.count])
        document = stack([editor_header, editor_frame], vertical=True, spacing=8)
        controls = stack([self.record, self.correct, kit.NSView.alloc().init(), self.copy])
        footer = stack([self.spinner, self.status], spacing=7)
        layout = stack([header, options, document, controls, footer], vertical=True, spacing=16)
        layout.setCustomSpacing_afterView_(22, header)
        layout.setCustomSpacing_afterView_(10, controls)

        content = self.window.contentView()
        layout.setTranslatesAutoresizingMaskIntoConstraints_(False)
        content.addSubview_(layout)
        guide = self.window.contentLayoutGuide()
        kit.NSLayoutConstraint.activateConstraints_([
            layout.leadingAnchor().constraintEqualToAnchor_constant_(guide.leadingAnchor(), 27),
            layout.trailingAnchor().constraintEqualToAnchor_constant_(guide.trailingAnchor(), -27),
            layout.topAnchor().constraintEqualToAnchor_constant_(guide.topAnchor(), 17),
            layout.bottomAnchor().constraintEqualToAnchor_constant_(guide.bottomAnchor(), -18),
            *[view.widthAnchor().constraintEqualToAnchor_(layout.widthAnchor())
              for view in (header, document, controls, footer)],
            editor_header.widthAnchor().constraintEqualToAnchor_(document.widthAnchor()),
            editor_frame.widthAnchor().constraintEqualToAnchor_(document.widthAnchor()),
            self.spinner.widthAnchor().constraintEqualToConstant_(16),
            self.spinner.heightAnchor().constraintEqualToConstant_(16),
        ])
        self.installMenus()
        self.refresh()
        self.window.center()
        self.window.makeKeyAndOrderFront_(None)
        kit.NSApp.activateIgnoringOtherApps_(True)

    @python_method
    def installMenus(self):
        bar = kit.NSMenu.alloc().init()
        menus = {
            "Skrybek": [("Ustawienia…", "showSettings:", ",", self),
                        ("Zakończ Skrybek", "terminate:", "q", None)],
            "Edycja": [("Cofnij", "undo:", "z", None), ("Ponów", "redo:", "Z", None),
                       ("Wytnij", "cut:", "x", None), ("Kopiuj zaznaczenie", "copy:", "c", None),
                       ("Wklej", "paste:", "v", None), ("Zaznacz wszystko", "selectAll:", "a", None)],
        }
        for name, commands in menus.items():
            item = kit.NSMenuItem.alloc().init()
            menu = kit.NSMenu.alloc().initWithTitle_(name)
            for title, action, key, target in commands:
                command = menu.addItemWithTitle_action_keyEquivalent_(title, action, key)
                command.setTarget_(target)
            item.setSubmenu_(menu)
            bar.addItem_(item)
        kit.NSApp.setMainMenu_(bar)

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True

    def applicationShouldTerminate_(self, app):
        return kit.NSTerminateNow if self.state == "idle" else kit.NSTerminateCancel

    def windowShouldClose_(self, window):
        return self.state == "idle"

    def optionsChanged_(self, sender):
        self.prefs.setObject_forKey_(self.backend.titleOfSelectedItem(), "backend")
        self.prefs.setBool_forKey_(bool(self.llm.state()), "correction")
        self.refresh()

    def textDidChange_(self, notification):
        self.refresh()

    @python_method
    def refresh(self):
        idle = self.state == "idle"
        recording = self.state == "recording"
        text = str(self.editor.string())
        length = len(text)
        unit = "znaków"
        if length == 1:
            unit = "znak"
        elif length % 10 in (2, 3, 4) and not 12 <= length % 100 <= 14:
            unit = "znaki"
        self.count.setStringValue_(f"{length:,}".replace(",", " ") + " " + unit)
        self.record.setEnabled_(idle or recording)
        self.record.setTitle_("Zakończ nagranie" if recording else "Rozpocznij nagranie")
        symbol = "stop.fill" if recording else "mic.fill"
        self.record.setImage_(kit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None))
        self.record.setBezelColor_(kit.NSColor.systemRedColor() if recording else kit.NSColor.systemTealColor())
        self.settings.setEnabled_(idle)
        self.backend.setEnabled_(idle)
        self.llm.setEnabled_(idle)
        self.editor.setEditable_(idle)
        self.copy.setEnabled_(bool(text.strip()))
        self.correct.setEnabled_(idle and bool(self.llm.state()) and bool(text.strip()))
        if idle:
            self.spinner.stopAnimation_(None)
        else:
            self.spinner.startAnimation_(None)

    def showSettings_(self, sender):
        if self.state != "idle" or self.window.attachedSheet():
            return
        alert = kit.NSAlert.alloc().init()
        alert.setMessageText_("Ustawienia Skrybka")
        alert.setInformativeText_("LiteRT wymaga lokalnego pliku .litertlm. Pusty prompt przywraca domyślną korektę.")
        alert.addButtonWithTitle_("Zapisz")
        alert.addButtonWithTitle_("Anuluj")
        form = kit.NSView.alloc().initWithFrame_(kit.NSMakeRect(0, 0, 420, 220))
        fields = {}
        for index, (key, title, _) in enumerate(PREFERENCES):
            y = 168 - index * 54
            caption = label(title, 12, weight=kit.NSFontWeightMedium)
            caption.setFrame_(kit.NSMakeRect(0, y + 26, 420, 17))
            form.addSubview_(caption)
            field = kit.NSTextField.alloc().initWithFrame_(kit.NSMakeRect(0, y, 332 if key == "model" else 420, 24))
            field.setStringValue_(self.prefs.stringForKey_(key) or "")
            field.setAccessibilityLabel_(title)
            form.addSubview_(field)
            fields[key] = field
        choose = button("Wybierz…", self, "chooseModel:")
        choose.setFrame_(kit.NSMakeRect(334, 60, 86, 24))
        form.addSubview_(choose)
        self.model_field = fields["model"]
        alert.setAccessoryView_(form)

        def finished(response):
            if response == kit.NSAlertFirstButtonReturn:
                for key, field in fields.items():
                    value = str(field.stringValue()).strip()
                    default = next(default for name, _, default in PREFERENCES if name == key)
                    self.prefs.setObject_forKey_(value or default, key)
                self.statusMessage("Zapisano ustawienia")

        alert.beginSheetModalForWindow_completionHandler_(self.window, finished)

    def chooseModel_(self, sender):
        picker = kit.NSOpenPanel.openPanel()
        picker.setAllowedFileTypes_(["litertlm"])
        picker.setAllowsMultipleSelection_(False)

        def picked(response):
            if response == kit.NSModalResponseOK:
                self.model_field.setStringValue_(picker.URL().path())

        picker.beginSheetModalForWindow_completionHandler_(sender.window(), picked)

    @python_method
    def statusMessage(self, message, error=False):
        self.status.setStringValue_(message)
        self.status.setToolTip_(message)
        self.status.setTextColor_(kit.NSColor.systemRedColor() if error else kit.NSColor.secondaryLabelColor())

    @python_method
    def correctionOptions(self):
        model = self.prefs.stringForKey_("model")
        if not model or not Path(model).expanduser().is_file():
            self.statusMessage("Wybierz plik modelu .litertlm w ustawieniach.", error=True)
            self.showSettings_(None)
            return None
        prompt = self.prefs.stringForKey_("prompt") or None
        try:
            resolve_litert_system_prompt(prompt)
        except (LiteRTError, OSError, ValueError) as error:
            self.statusMessage(str(error), error=True)
            return None
        return str(Path(model).expanduser()), prompt

    def recordAction_(self, sender):
        if self.state == "recording":
            self.stop.set()
            self.state = "stopping"
            self.statusMessage("Kończenie nagrania…")
            self.refresh()
            return
        if self.state != "idle" or self.window.attachedSheet():
            return
        correction = self.correctionOptions() if self.llm.state() else None
        if self.llm.state() and correction is None:
            return
        self.pending_correction = correction
        authorization = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
        if authorization == AVAuthorizationStatusNotDetermined:
            self.state = "permission"
            self.statusMessage("Oczekiwanie na dostęp do mikrofonu…")
            self.refresh()
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeAudio, lambda ok: self.post("permissionResolved:", bool(ok))
            )
        else:
            self.permissionResolved_(authorization == AVAuthorizationStatusAuthorized)

    def permissionResolved_(self, allowed):
        if not allowed:
            self.state = "idle"
            self.statusMessage("Zezwól Skrybkowi na użycie mikrofonu w Ustawieniach systemowych.", error=True)
            self.refresh()
            return
        self.stop.clear()
        self.state = "recording"
        self.statusMessage("Nagrywanie · Zatrzymanie po 3 s ciszy lub klawiszem Esc")
        self.refresh()
        self.worker.submit(self.recordAndTranscribe, self.backend.titleOfSelectedItem().lower(),
                           self.prefs.stringForKey_("whisper"), self.prefs.stringForKey_("parakeet"),
                           self.pending_correction)

    @python_method
    def recordAndTranscribe(self, backend, whisper, parakeet, correction):
        try:
            audio = record_until_silence(stop_event=self.stop)
            if not audio.size:
                raise ValueError("Nagranie jest puste. Spróbuj ponownie.")
            self.post("phaseChanged:", "Rozpoznawanie mowy…")
            with TemporaryDirectory(prefix="skrybek-") as directory:
                path = str(Path(directory) / "recording.wav")
                save_wav_pcm16(path, 16000, audio)
                transcript = transcribe_audio(path, backend, whisper, parakeet)
        except Exception as error:
            self.post("jobFailed:", str(error))
            return
        if correction:
            self.post("phaseChanged:", "Korekta tekstu przez LiteRT…")
            self.correctText(transcript, correction, fallback=transcript)
        else:
            self.post("jobFinished:", transcript)

    def correctAction_(self, sender):
        if self.state != "idle" or not self.llm.state():
            return
        text = str(self.editor.string())
        correction = self.correctionOptions() if text.strip() else None
        if correction:
            self.phaseChanged_("Korekta tekstu przez LiteRT…")
            self.worker.submit(self.correctText, text, correction)

    @python_method
    def correctText(self, text, options, fallback=None):
        try:
            result = process_text_with_litert_subprocess(text, *options)
        except Exception as error:
            self.post("correctionFailed:", (fallback, str(error)))
        else:
            self.post("jobFinished:", result)

    @python_method
    def post(self, selector, value):
        self.performSelectorOnMainThread_withObject_waitUntilDone_(selector, value, False)

    def phaseChanged_(self, message):
        self.state = "processing"
        self.statusMessage(message)
        self.refresh()

    def jobFinished_(self, text):
        self.state = "idle"
        self.editor.setEditable_(True)
        old_range = NSMakeRange(0, self.editor.textStorage().length())
        self.editor.insertText_replacementRange_(text, old_range)
        self.editor.setSelectedRange_(NSMakeRange(0, 0))
        self.editor.scrollRangeToVisible_(NSMakeRange(0, 0))
        self.statusMessage("Tekst gotowy do edycji")
        self.refresh()
        self.window.makeFirstResponder_(self.editor)

    def jobFailed_(self, error):
        self.state = "idle"
        self.statusMessage(error, error=True)
        self.refresh()

    def correctionFailed_(self, result):
        fallback, error = result
        if fallback is not None:
            self.jobFinished_(fallback)
        self.jobFailed_(f"Błąd korekty LiteRT · {error}")

    def copyAction_(self, sender):
        text = str(self.editor.string())
        if text.strip():
            if copy_to_clipboard(text):
                if self.state == "idle":
                    self.statusMessage("Skopiowano tekst do schowka")
            else:
                self.statusMessage("Nie udało się skopiować tekstu.", error=True)

    @python_method
    def handleEvent(self, event):
        if event.keyCode() == 53 and self.state == "recording" and event.window() == self.window:
            self.recordAction_(None)
            return None
        return event


def run():
    app = kit.NSApplication.sharedApplication()
    app.setActivationPolicy_(kit.NSApplicationActivationPolicyRegular)
    delegate = Desktop.alloc().init()
    app.setDelegate_(delegate)
    monitor = kit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
        kit.NSEventMaskKeyDown, delegate.handleEvent
    )
    try:
        app.run()
    finally:
        kit.NSEvent.removeMonitor_(monitor)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
