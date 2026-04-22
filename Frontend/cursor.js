const cursorDot = document.createElement('div');
cursorDot.classList.add('cursor-dot');
document.body.appendChild(cursorDot);

const cursorOutline = document.createElement('div');
cursorOutline.classList.add('cursor-outline');
document.body.appendChild(cursorOutline);

window.addEventListener('mousemove', function(e) {
    const posX = e.clientX;
    const posY = e.clientY;

    cursorDot.style.left = `${posX}px`;
    cursorDot.style.top = `${posY}px`;

    // The outline animates towards the mouse for a smooth trailing effect
    cursorOutline.animate({
        left: `${posX}px`,
        top: `${posY}px`
    }, { duration: 500, fill: "forwards" });
});

// Add hover effect to all links, buttons, and textareas
document.querySelectorAll('a, button, textarea').forEach(el => {
    el.addEventListener('mouseenter', () => {
        cursorOutline.classList.add('hovering');
    });
    el.addEventListener('mouseleave', () => {
        cursorOutline.classList.remove('hovering');
    });
});
