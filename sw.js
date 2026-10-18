/* Service worker for the Tech Journal.
 *
 * GitHub Pages serves every file with `Cache-Control: max-age=600` behind a
 * multi-node CDN. A normal reload (F5) therefore flaps between stale and fresh
 * copies depending on which edge node answers, while a hard reload (Ctrl+F5)
 * bypasses the caches. Docsify's `cache-control: no-cache` request header does
 * not help either, because the edge ignores client cache directives.
 *
 * The only reliable escape is to request a URL the cache has never seen, so
 * every fetch below is rewritten with a unique query string. This bypasses the
 * browser cache and the edge cache at once, so reloads always show the latest
 * version. Bump SW_VERSION to force the browser to re-register this worker.
 */

var SW_VERSION = '2026-10-18-1';

self.addEventListener('install', function () {
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', function (event) {
  var request = event.request;

  if (request.method !== 'GET') {
    return;
  }

  var url;
  try {
    url = new URL(request.url);
  } catch (e) {
    return;
  }

  // Leave cross-origin requests (fonts, CDN scripts, KaTeX, ...) untouched.
  if (url.origin !== self.location.origin) {
    return;
  }

  var separator = url.search ? '&' : '?';
  var busted = url.href + separator + '_sw=' + SW_VERSION + '-' + Date.now();

  event.respondWith(
    fetch(busted, { cache: 'no-store', credentials: 'same-origin' }).catch(
      function () {
        return fetch(request);
      }
    )
  );
});
