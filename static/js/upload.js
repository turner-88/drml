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

  function show(file) {
    nameEl.textContent = file.name;
    sizeEl.textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';
    var reader = new FileReader();
    reader.onload = function (ev) {
      previewImg.src = ev.target.result;
      setState(true);
    };
    reader.readAsDataURL(file);
  }

  input.addEventListener('change', function () {
    if (input.files.length) show(input.files[0]);
  });

  document.getElementById('dz-clear').addEventListener('click', function () {
    input.value = '';
    previewImg.removeAttribute('src');
    setState(false);
  });

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

  setState(false);
})();
