/**
 * Client-side image compression utility.
 * Uses browser-image-compression (loaded via CDN) to resize and compress
 * images before form submission, reducing upload size and bandwidth.
 *
 * Usage: add @change="compressFileInput($event.target, { maxSizeMB: 0.5, maxWidthOrHeight: 800 })"
 * to any <input type="file"> element that accepts images.
 */

// Preset compression options per upload context.
var COMPRESS_THUMBNAIL = { maxSizeMB: 0.5, maxWidthOrHeight: 800 };
var COMPRESS_FEATURED  = { maxSizeMB: 1.0, maxWidthOrHeight: 1920 };
var COMPRESS_GALLERY   = { maxSizeMB: 0.8, maxWidthOrHeight: 1600 };
var COMPRESS_ATTACH    = { maxSizeMB: 1.0, maxWidthOrHeight: 1200 };
var COMPRESS_LOGO      = { maxSizeMB: 0.5, maxWidthOrHeight: 800 };

/**
 * Compresses image files in a file input and replaces them in-place
 * using the DataTransfer API. Non-image files (e.g. PDFs) pass through
 * unchanged.
 *
 * @param {HTMLInputElement} input - The file input element.
 * @param {Object} opts - Options passed to imageCompression().
 * @param {number} opts.maxSizeMB - Target max file size in MB.
 * @param {number} opts.maxWidthOrHeight - Max pixel dimension.
 * @returns {Promise<void>}
 */
async function compressFileInput(input, opts) {
    if (typeof imageCompression === 'undefined') return;
    if (!input.files || input.files.length === 0) return;

    var options = {
        maxSizeMB: opts.maxSizeMB || 1,
        maxWidthOrHeight: opts.maxWidthOrHeight || 1920,
        useWebWorker: true,
        preserveExif: false
    };

    var dt = new DataTransfer();

    for (var i = 0; i < input.files.length; i++) {
        var file = input.files[i];

        // Only compress image files; pass through PDFs and others.
        if (!file.type.startsWith('image/')) {
            dt.items.add(file);
            continue;
        }

        try {
            var compressed = await imageCompression(file, options);
            // Keep original name for server-side MIME detection.
            var out = new File([compressed], file.name, { type: compressed.type });
            dt.items.add(out);
        } catch (e) {
            // Compression failed — keep the original file.
            dt.items.add(file);
        }
    }

    input.files = dt.files;
}
