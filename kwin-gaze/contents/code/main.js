// Send the global cursor position to the jelly clock (org.tiger.Clock) over D-Bus,
// at most ~30 times a second and only while the mouse moves. Nothing runs at rest.
var last = 0;
var pending = false;
var timer = new QTimer();
timer.singleShot = true;
timer.timeout.connect(send);

function send() {
    pending = false;
    last = Date.now();
    var p = workspace.cursorPos;
    callDBus("org.tiger.Clock", "/org/tiger/Clock", "org.tiger.Clock", "Look", Math.round(p.x), Math.round(p.y));
}

workspace.cursorPosChanged.connect(function () {
    var wait = 33 - (Date.now() - last);
    if (wait <= 0) {
        send();
    } else if (!pending) {            // trailing update: the final resting position always arrives
        pending = true;
        timer.interval = wait;
        timer.start();
    }
});
