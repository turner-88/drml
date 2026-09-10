// Drop-zone behaviour for the new-scan form. Deliberately dependency-free: the
// rest of the UI ships no JavaScript framework, and this is the only page that
// needs client-side state.
(function () {
  var form = document.getElementById('scan-form');
  if (!form) return;

  var input = document.getElementById('scan-image');
  var zone = document.getElementById('dropzone');
  var emptyState = document.getElementById('dz-empty');
  var previewState = document.getElementById('dz-preview');
  var previewImg = document.getElementById('dz-img');
  var previewFrame = document.getElementById('dz-frame');
  var previewDoc = document.getElementById('dz-doc');
  var modeField = document.getElementById('scan-mode');
  var modeTitle = document.getElementById('mode-title');
  var modeNote = document.getElementById('mode-note');
  var iconPDF = document.getElementById('dz-icon-pdf');
  var iconImage = document.getElementById('dz-icon-image');
  var emptyText = document.getElementById('dz-empty-text');
  var emptyHint = document.getElementById('dz-empty-hint');
  var analysisNote = document.getElementById('analysis-note');
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.mode-tab'));
  var nameEl = document.getElementById('dz-name');
  var sizeEl = document.getElementById('dz-size');
  var submit = document.getElementById('scan-submit');
  var submitIdle = document.getElementById('submit-idle');
  var submitBusy = document.getElementById('submit-busy');

  // The two states are toggled with [hidden] rather than being added to and
  // removed from the DOM. The file input lives outside both of them for the
  // same reason: if it were inside the block that disappears once a file is
  // chosen, the form would submit with no "image" field at all.
  function setState(hasFile) {
    emptyState.hidden = hasFile;
    previewState.hidden = !hasFile;
    submit.disabled = !hasFile;
  }

  // Each mode differs only in what the file input accepts and what the copy
  // says; the form, the field name and the submit path are shared.
  var MODES = {
    image: {
      accept: 'image/jpeg,image/png,image/webp',
      title: 'Unggah Foto Fundus',
      note: 'Format JPG, PNG, atau WebP — maks. 12 MB',
      empty: 'Tarik & lepas file gambar ke sini, atau',
      hint: 'Khusus foto kamera fundus retina. Foto biasa atau tangkapan layar otomatis ditolak.',
      analysis: 'Mohon jangan menutup atau memuat ulang halaman ini selama proses analisis berlangsung.'
    },
    pdf: {
      accept: 'application/pdf',
      title: 'Unggah PDF IMAGEnet',
      note: 'Hasil ekspor PDF dari IMAGEnet — maks. 32 MB',
      empty: 'Tarik & lepas file PDF ke sini, atau',
      hint: 'Foto fundus diambil otomatis dari PDF. Laporan dua mata menghasilkan dua pemeriksaan (OD dan OS).',
      analysis: 'Analisis PDF dua mata memerlukan waktu lebih lama. Mohon jangan menutup atau memuat ulang halaman ini.'
    }
  };

  function isPDF(file) {
    return file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
  }

  function setMode(mode) {
    // Normalise rather than fall through to a default config: leaving the
    // posted field on a mode that does not exist would confuse the server.
    if (!MODES[mode]) mode = 'pdf';
    var cfg = MODES[mode];
    modeField.value = mode;
    input.setAttribute('accept', cfg.accept);
    modeTitle.textContent = cfg.title;
    modeNote.textContent = cfg.note;
    emptyText.textContent = cfg.empty;
    emptyHint.textContent = cfg.hint;
    analysisNote.textContent = cfg.analysis;
    iconPDF.hidden = mode !== 'pdf';
    iconImage.hidden = mode === 'pdf';
    tabs.forEach(function (t) {
      var on = t.dataset.mode === mode;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
  }

  function clearFile() {
    input.value = '';
    previewImg.removeAttribute('src');
    setState(false);
  }

  function show(file) {
    nameEl.textContent = file.name;
    sizeEl.textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';

    // Follow the dropped file rather than the selected tab: someone who drags
    // a PDF onto the image tab meant to screen a PDF.
    var pdf = isPDF(file);
    if (pdf !== (modeField.value === 'pdf')) setMode(pdf ? 'pdf' : 'image');

    previewFrame.hidden = pdf;
    previewDoc.hidden = !pdf;
    if (pdf) {
      previewImg.removeAttribute('src');
      setState(true);
      return;
    }
    var reader = new FileReader();
    reader.onload = function (ev) {
      previewImg.src = ev.target.result;
      setState(true);
    };
    reader.readAsDataURL(file);
  }

  tabs.forEach(function (t) {
    t.addEventListener('click', function () {
      if (modeField.value === t.dataset.mode) return;
      setMode(t.dataset.mode);
      // The chosen file belongs to the old mode, so drop it rather than
      // submitting a PDF as an image.
      clearFile();
    });
  });

  input.addEventListener('change', function () {
    if (input.files.length) show(input.files[0]);
  });

  document.getElementById('dz-clear').addEventListener('click', clearFile);

  ['dragenter', 'dragover'].forEach(function (evt) {
    zone.addEventListener(evt, function (e) {
      e.preventDefault();
      zone.classList.add('is-dragging');
    });
  });
  ['dragleave', 'dragend'].forEach(function (evt) {
    zone.addEventListener(evt, function (e) {
      e.preventDefault();
      zone.classList.remove('is-dragging');
    });
  });

  zone.addEventListener('drop', function (e) {
    e.preventDefault();
    zone.classList.remove('is-dragging');
    if (!e.dataTransfer.files.length) return;
    // Assign to the real input so the file is part of the form submission.
    input.files = e.dataTransfer.files;
    show(e.dataTransfer.files[0]);
  });

  form.addEventListener('submit', function () {
    submit.disabled = true;
    submitIdle.hidden = true;
    submitBusy.hidden = false;
  });

  setMode(modeField.value);
  setState(false);
})();
