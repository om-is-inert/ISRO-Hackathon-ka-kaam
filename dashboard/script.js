// Initialize Feather Icons
feather.replace();

// ==========================================
// FLOATING NAV TRACKING
// ==========================================
const sections = document.querySelectorAll('.observer-section');
const navDots = document.querySelectorAll('.nav-dot');
const navIndicator = document.getElementById('nav-indicator');

const navObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        // High threshold so it switches when section is prominent
        if (entry.isIntersecting) {
            const currentId = entry.target.getAttribute('id');
            const targetDot = document.querySelector(`.nav-dot[data-section="${currentId}"]`);
            if (targetDot && navIndicator) {
                // Center the 16px pill over the 6px dot
                const dotTop = targetDot.offsetTop;
                navIndicator.style.transform = `translateY(${dotTop - 5}px)`;
            }
        }
    });
}, { threshold: 0.3, rootMargin: "-10% 0px -40% 0px" });

sections.forEach(sec => navObserver.observe(sec));

navDots.forEach(dot => {
    dot.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = dot.getAttribute('data-section');
        document.getElementById(targetId).scrollIntoView({ behavior: 'smooth' });
    });
});

// ==========================================
// TYPEWRITER EFFECT
// ==========================================
const subtitleText = "We taught a machine to see time.";
const subtitleEl = document.getElementById('typewriter-text');
let typeIndex = 0;

function typeWriter() {
    if (subtitleEl && typeIndex < subtitleText.length) {
        subtitleEl.innerHTML += subtitleText.charAt(typeIndex);
        typeIndex++;
        // Speed up slightly towards end
        setTimeout(typeWriter, 40 + Math.random() * 40);
    }
}
// Start after initial page load reveals
setTimeout(typeWriter, 600);

// ==========================================
// HERO STARFIELD (CANVAS)
// ==========================================
const canvas = document.getElementById('hero-canvas');
const heroSection = document.getElementById('hero');

if (canvas && heroSection) {
    const ctx = canvas.getContext('2d');
    let stars = [];
    let shootingStars = [];
    let time = 0;
    
    let isMobile = window.innerWidth < 768;
    const NUM_STARS = isMobile ? 150 : 300;
    
    const orbit = {
        centerX: 0,
        centerY: 0,
        radiusX: 0,
        radiusY: 0,
        angle: 0,
        speed: 0.0004,
        trailLength: 28
    };
    const trailPoints = [];
    
    function initCanvas() {
        const dpr = window.devicePixelRatio || 1;
        isMobile = window.innerWidth < 768;
        
        canvas.width = heroSection.offsetWidth * dpr;
        canvas.height = heroSection.offsetHeight * dpr;
        canvas.style.width = heroSection.offsetWidth + 'px';
        canvas.style.height = heroSection.offsetHeight + 'px';
        
        ctx.scale(dpr, dpr);
        
        // Setup Orbit parameters
        orbit.centerX = heroSection.offsetWidth * 0.5;
        orbit.centerY = heroSection.offsetHeight * 0.5;
        orbit.radiusX = isMobile ? heroSection.offsetWidth * 0.35 : heroSection.offsetWidth * 0.28;
        orbit.radiusY = isMobile ? heroSection.offsetHeight * 0.12 : heroSection.offsetHeight * 0.18;
        
        // Clear trail
        trailPoints.length = 0;
        
        // Re-init stars
        stars = [];
        for (let i = 0; i < (isMobile ? 150 : 300); i++) {
            stars.push({
                x: Math.random() * heroSection.offsetWidth,
                y: Math.random() * heroSection.offsetHeight,
                radius: Math.random() * 1.4 + 0.3,
                baseOpacity: Math.random() * 0.5 + 0.3,
                twinkleSpeed: Math.random() * 0.8 + 0.3,
                twinkleOffset: Math.random() * Math.PI * 2,
                glowStar: Math.random() < 0.12
            });
        }
    }
    
    window.addEventListener('resize', initCanvas);
    initCanvas();
    
    function spawnShootingStar() {
        if (isMobile) return; // No shooting stars on mobile
        
        const isTop = Math.random() > 0.5;
        const x = isTop ? Math.random() * heroSection.offsetWidth : heroSection.offsetWidth;
        const y = isTop ? 0 : Math.random() * heroSection.offsetHeight * 0.5;
        
        const angle = (210 + Math.random() * 20) * (Math.PI / 180);
        const speed = 7;
        
        shootingStars.push({
            x: x,
            y: y,
            tailX: Math.cos(angle) * 90,
            tailY: Math.sin(angle) * 90,
            speedX: Math.cos(angle) * speed,
            speedY: Math.sin(angle) * speed,
            life: 84, // 1.4 seconds at 60fps
            maxLife: 84
        });
        
        setTimeout(spawnShootingStar, Math.random() * 4000 + 6000); // 6-10 seconds
    }
    if (!isMobile) {
        setTimeout(spawnShootingStar, Math.random() * 4000 + 6000);
    }
    
    let lastTime = performance.now();
    let animId;
    
    function draw(timestamp) {
        const deltaTime = timestamp - lastTime;
        lastTime = timestamp;
        
        // Guard against huge delta times
        const safeDeltaTime = Math.min(deltaTime, 32); 
        
        ctx.clearRect(0, 0, heroSection.offsetWidth, heroSection.offsetHeight);
        
        // 1. Orbit path
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 0.8;
        ctx.setLineDash([3, 10]);
        ctx.beginPath();
        ctx.ellipse(
            orbit.centerX, orbit.centerY,
            orbit.radiusX, orbit.radiusY,
            0, 0, Math.PI * 2
        );
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        
        // 2. Stars
        stars.forEach(star => {
            const twinkle = Math.sin(time * star.twinkleSpeed + star.twinkleOffset);
            const opacity = star.baseOpacity + twinkle * 0.2;
            
            ctx.save();
            
            if (star.glowStar) {
                const glow = ctx.createRadialGradient(
                    star.x, star.y, 0,
                    star.x, star.y, star.radius * 5
                );
                glow.addColorStop(0, `rgba(255,255,255,${opacity * 0.4})`);
                glow.addColorStop(1, 'rgba(255,255,255,0)');
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(star.x, star.y, star.radius * 5, 0, Math.PI * 2);
                ctx.fill();
            }
            
            ctx.globalAlpha = Math.max(0, Math.min(1, opacity));
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.restore();
        });
        
        // 3. Shooting stars
        for (let i = shootingStars.length - 1; i >= 0; i--) {
            const s = shootingStars[i];
            s.x += s.speedX;
            s.y += s.speedY;
            s.life--;
            
            if (s.life <= 0) {
                shootingStars.splice(i, 1);
                continue;
            }
            
            s.opacity = s.life / s.maxLife;
            s.progress = Math.min(1, (s.maxLife - s.life) / 10);
            
            const gradient = ctx.createLinearGradient(
                s.x, s.y,
                s.x - s.tailX, s.y - s.tailY
            );
            gradient.addColorStop(0, `rgba(255,255,255,${s.opacity * 0.9})`);
            gradient.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(s.x - s.tailX * s.progress, s.y - s.tailY * s.progress);
            ctx.stroke();
        }
        
        // 4. Satellite trail
        orbit.angle += orbit.speed * safeDeltaTime;
        const satX = orbit.centerX + Math.cos(orbit.angle) * orbit.radiusX;
        const satY = orbit.centerY + Math.sin(orbit.angle) * orbit.radiusY;
        
        trailPoints.push({ x: satX, y: satY });
        if (trailPoints.length > orbit.trailLength) trailPoints.shift();
        
        trailPoints.forEach((point, i) => {
            const progress = i / trailPoints.length;
            ctx.save();
            ctx.globalAlpha = progress * 0.35;
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(point.x, point.y, 1.2 * progress, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });
        
        // 5. Satellite dot
        const satGlow = ctx.createRadialGradient(satX, satY, 0, satX, satY, 8);
        satGlow.addColorStop(0, 'rgba(255,255,255,0.6)');
        satGlow.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = satGlow;
        ctx.beginPath();
        ctx.arc(satX, satY, 8, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = '#ffffff';
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.arc(satX, satY, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
        
        time += 0.016;
        animId = requestAnimationFrame(draw);
    }
    
    const heroObserver = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) {
            if (!animId) {
                lastTime = performance.now();
                animId = requestAnimationFrame(draw);
            }
        } else {
            if (animId) cancelAnimationFrame(animId);
            animId = null;
        }
    }, { threshold: 0.1 });
    heroObserver.observe(heroSection);
}

// ==========================================
// SCROLL ANIMATIONS
// ==========================================
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            
            if (entry.target.querySelector('.counter') && !entry.target.hasAttribute('data-counted')) {
                entry.target.setAttribute('data-counted', 'true');
                animateCounters(entry.target);
            }
        } else {
            entry.target.classList.remove('visible');
        }
    });
}, { 
    threshold: 0.12,
    rootMargin: '0px 0px -60px 0px'
});

document.querySelectorAll(
  '.reveal-element, .section-heading, .metric-card, .architecture-node, .training-card, .team-card'
).forEach(el => observer.observe(el));

document.querySelectorAll('.section-heading').forEach(heading => {
  const words = heading.textContent.trim().split(' ');
  heading.innerHTML = words.map((word, i) => 
    `<span class="word-wrap" style="display:inline-block; overflow:hidden; vertical-align:bottom">
      <span class="word-inner" style="
        display:inline-block;
        transform: translateY(100%);
        transition: transform ${600 + i * 80}ms cubic-bezier(0.16, 1, 0.3, 1);
        transition-delay: ${i * 60}ms;
      ">${word}</span>
    </span> `
  ).join('');
});

const headingObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    const inners = entry.target.querySelectorAll('.word-inner');
    if (entry.isIntersecting) {
      inners.forEach(w => w.style.transform = 'translateY(0)');
    } else {
      inners.forEach(w => w.style.transform = 'translateY(100%)');
    }
  });
}, { threshold: 0.2 });

document.querySelectorAll('.section-heading').forEach(h => headingObserver.observe(h));

// ==========================================
// NUMBER COUNT-UP LOGIC (easeOutExpo)
// ==========================================
function animateCounters(container) {
    const counters = container.querySelectorAll('.counter');
    const duration = 2000;

    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-target'));
        const decimals = parseInt(counter.getAttribute('data-decimals') || 0);
        let startTime = null;
        
        const updateCount = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = timestamp - startTime;
            const percentage = Math.min(progress / duration, 1);
            
            // easeOutExpo
            const easeOut = percentage === 1 ? 1 : 1 - Math.pow(2, -10 * percentage);
            
            const currentValue = target * easeOut;
            counter.innerText = currentValue.toFixed(decimals);
            
            if (progress < duration) {
                requestAnimationFrame(updateCount);
            } else {
                counter.innerText = target.toFixed(decimals);
            }
        };
        
        requestAnimationFrame(updateCount);
    });
}

// ==========================================
// IMAGE COMPARISON SLIDER
// ==========================================
const sliderInput = document.getElementById('slider-input');
const sliderForeground = document.getElementById('slider-foreground');
const sliderHandle = document.getElementById('slider-handle');

if (sliderInput && sliderForeground && sliderHandle) {
    const initVal = sliderInput.value;
    sliderForeground.style.clipPath = `polygon(0 0, ${initVal}% 0, ${initVal}% 100%, 0 100%)`;
    sliderHandle.style.left = `${initVal}%`;

    sliderInput.addEventListener('input', (e) => {
        const value = e.target.value;
        sliderForeground.style.clipPath = `polygon(0 0, ${value}% 0, ${value}% 100%, 0 100%)`;
        sliderHandle.style.left = `${value}%`;
        sliderHandle.classList.remove('pulse-hint'); // Remove hint on interaction
    });
}

// ==========================================
// ARCHITECTURE TOOLTIPS (MOBILE TOGGLE)
// ==========================================
const nodeWrappers = document.querySelectorAll('.node-wrapper');
nodeWrappers.forEach(node => {
    node.addEventListener('click', () => {
        nodeWrappers.forEach(n => { if(n !== node) n.classList.remove('active'); });
        node.classList.toggle('active');
    });
});

// ==========================================
// CHART.JS SETUP
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    const labels = Array.from({length: 71}, (_, i) => i);
    
    
    const psnrRefined = [36.69, 38.84, 33.81, 16.47, 20.81, 24.03, 33.34, 36.28, 19.91, 19.99, 19.34, 33.43, 13.71, 44.41, 39.48, 40.09, 34.95, 42.07, 21.43, 46.24, 25.13, 40.44, 22.56, 45.54, 32.89, 19.77, 25.63, 33.49, 36.56, 33.19, 34.21, 27.94, 30.91, 41.54, 33.18, 26.18, 29.01, 14.29, 37.47, 45.35, 21.51, 21.66, 27.81, 26.28, 44.36, 38.01, 32.43, 36.72, 20.78, 24.03, 20.44, 27.37, 24.23, 27.29, 28.62, 28.09, 22.18, 24.0, 33.54, 32.89, 26.1, 28.38, 29.98, 32.93, 27.35, 30.78, 22.61, 18.95, 20.85, 20.64];
    const psnrCoarse = [28.08, 33.41, 29.49, 17.5, 18.89, 20.71, 29.04, 31.53, 19.94, 20.41, 19.24, 33.18, 13.26, 43.9, 34.09, 34.96, 35.24, 41.81, 20.57, 42.14, 23.4, 32.99, 21.57, 43.62, 30.47, 19.22, 25.54, 23.69, 35.93, 31.9, 34.41, 27.76, 26.48, 34.91, 31.88, 38.83, 29.76, 14.4, 32.72, 43.68, 22.02, 20.88, 26.89, 24.14, 44.18, 37.66, 28.98, 31.94, 20.77, 20.71, 20.08, 29.28, 23.66, 26.92, 29.51, 28.66, 21.53, 23.46, 32.54, 30.47, 25.31, 40.73, 23.96, 29.42, 27.51, 30.34, 21.93, 18.71, 20.31, 19.81];
    const ssimRefined = [0.9687, 0.9814, 0.9702, 0.911, 0.8796, 0.9043, 0.9807, 0.9654, 0.7381, 0.7064, 0.5382, 0.957, 0.6878, 0.9922, 0.9927, 0.9944, 0.9718, 0.9932, 0.5431, 0.9934, 0.8863, 0.9808, 0.7499, 0.9895, 0.9586, 0.6024, 0.8783, 0.9929, 0.9862, 0.9307, 0.9455, 0.8845, 0.9763, 0.9931, 0.9891, 0.9859, 0.9641, 0.651, 0.9772, 0.9948, 0.669, 0.9249, 0.9516, 0.9536, 0.9943, 0.9692, 0.9699, 0.992, 0.8096, 0.9043, 0.7684, 0.9658, 0.8519, 0.8958, 0.9842, 0.9269, 0.7076, 0.9241, 0.9857, 0.9586, 0.9737, 0.9827, 0.9658, 0.9591, 0.9358, 0.9623, 0.9207, 0.7686, 0.6399, 0.8051];
    const ssimCoarse = [0.9629, 0.9716, 0.9601, 0.8832, 0.8706, 0.8885, 0.9691, 0.9591, 0.73, 0.6839, 0.5079, 0.9521, 0.6585, 0.9915, 0.9819, 0.9924, 0.9595, 0.992, 0.4825, 0.9932, 0.8188, 0.9771, 0.7177, 0.9896, 0.9535, 0.5594, 0.8706, 0.9713, 0.9834, 0.9247, 0.9412, 0.8797, 0.9645, 0.9854, 0.985, 0.9844, 0.9562, 0.6439, 0.9615, 0.9942, 0.6668, 0.8959, 0.9447, 0.9408, 0.9938, 0.9661, 0.9586, 0.9913, 0.7923, 0.8885, 0.737, 0.9618, 0.8387, 0.8887, 0.9786, 0.9227, 0.684, 0.8825, 0.9656, 0.9535, 0.9641, 0.9823, 0.9462, 0.9438, 0.9323, 0.9506, 0.8878, 0.7439, 0.5837, 0.8023];
    new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Refined (PSNR)',
                    data: psnrRefined,
                    borderColor: '#111111',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0,
                    yAxisID: 'y',
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointBackgroundColor: '#111111',
                    fill: false
                },
                {
                    label: 'Coarse (PSNR)',
                    data: psnrCoarse,
                    borderColor: '#888888',
                    borderDash: [4, 4],
                    borderWidth: 2,
                    tension: 0,
                    yAxisID: 'y',
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointBackgroundColor: '#888888',
                    fill: false
                },
                {
                    label: 'Refined (SSIM)',
                    data: ssimRefined,
                    borderColor: '#111111',
                    borderWidth: 2,
                    tension: 0,
                    yAxisID: 'y1',
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointBackgroundColor: '#111111',
                    fill: false
                },
                {
                    label: 'Coarse (SSIM)',
                    data: ssimCoarse,
                    borderColor: '#888888',
                    borderDash: [4, 4],
                    borderWidth: 2,
                    tension: 0,
                    yAxisID: 'y1',
                    pointBackgroundColor: '#888888',
                    fill: false
                }
            ]
        },
        options: {
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            },
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false, // Ensures vertical crosshair line across datasets
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    alert(`Selected Frame Index: ${index}`);
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        usePointStyle: true,
                        boxWidth: 6,
                        boxHeight: 6,
                        padding: 20,
                        font: { family: "'Inter', sans-serif", size: 13, weight: '500' },
                        color: '#111111'
                    }
                },
                tooltip: {
                    backgroundColor: '#111111',
                    titleFont: { family: "'Inter', sans-serif", size: 14, weight: '600' },
                    bodyFont: { family: "'Inter', sans-serif", size: 13 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: true,
                    boxPadding: 4
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    title: {
                        display: true,
                        text: 'Frame Index (0 - 70)',
                        font: { family: "'Inter', sans-serif", size: 13, weight: '500' },
                        color: '#666666'
                    },
                    ticks: {
                        font: { family: "'Inter', sans-serif", size: 12 },
                        color: '#666666',
                        maxTicksLimit: 15
                    },
                    border: { display: false }
                },
                'y': {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 10,
                    max: 50,
                    title: {
                        display: true,
                        text: 'PSNR (dB)',
                        font: { family: "'Inter', sans-serif", size: 13, weight: '500' },
                        color: '#111111'
                    },
                    grid: { color: '#F5F5F5' },
                    ticks: { font: { family: "'Inter', sans-serif", size: 12 }, color: '#666666' },
                    border: { display: false }
                },
                'y1': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0.4,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'SSIM',
                        font: { family: "'Inter', sans-serif", size: 13, weight: '500' },
                        color: '#111111'
                    },
                    grid: { drawOnChartArea: false },
                    border: { display: false }
                }
            }
        }
    });
});

// ==========================================
// GSAP STAGGERED MOBILE MENU
// ==========================================
const menuBtn = document.getElementById('mobile-menu-btn');
const menuContainer = document.getElementById('mobile-menu');
const layer1 = document.querySelector('.menu-underlay.layer-1');
const layer2 = document.querySelector('.menu-underlay.layer-2');
const menuPanel = document.querySelector('.menu-panel');
const menuTexts = document.querySelectorAll('.menu-text');
const menuNums = document.querySelectorAll('.menu-num');
const btnIcon = document.querySelector('.menu-btn-icon svg');
const textInner = document.getElementById('menu-text-inner');
const menuItems = document.querySelectorAll('.menu-nav .menu-item');

let isMenuOpen = false;

// Clone texts for rolling effect (3 cycles = 6 items total)
if (textInner) {
    const originalHTML = textInner.innerHTML;
    textInner.innerHTML = originalHTML + originalHTML + originalHTML;
}

function toggleMenu() {
    isMenuOpen = !isMenuOpen;
    
    if (isMenuOpen) {
        menuContainer.style.pointerEvents = 'auto';
        
        // Icon rotate to X (225deg)
        gsap.to(btnIcon, { rotation: 225, duration: 0.5, ease: "power3.inOut" });
        
        // Text rapid cycle down to last "Close" (index 5 out of 6 items)
        gsap.to(textInner, { 
            yPercent: -(5/6) * 100, 
            duration: 0.8, 
            ease: "power4.inOut" 
        });
        
        // Open Timeline
        const tl = gsap.timeline();
        
        // Underlays slide in
        tl.to([layer1, layer2], {
            x: "0%",
            duration: 0.5,
            ease: "power4.out",
            stagger: 0.07
        }, 0);
        
        // White panel slides in
        tl.to(menuPanel, {
            x: "0%",
            duration: 0.65,
            ease: "power4.out"
        }, 0.1);
        
        // Menu Labels (and their child superscripts)
        tl.fromTo(menuTexts, 
            { yPercent: 140, rotation: 10 },
            { yPercent: 0, rotation: 0, duration: 1, ease: "power4.out", stagger: 0.1 },
            0.2
        );
        
    } else {
        menuContainer.style.pointerEvents = 'none';
        
        // Icon reverse
        gsap.to(btnIcon, { rotation: 0, duration: 0.5, ease: "power3.inOut" });
        
        // Text rapid cycle back up to first "Menu" (index 0)
        gsap.to(textInner, { 
            yPercent: 0, 
            duration: 0.8, 
            ease: "power4.inOut" 
        });
        
        // Close Timeline
        const tl = gsap.timeline();
        
        // All layers slide back simultaneously
        tl.to([layer1, layer2, menuPanel], {
            x: "100%",
            duration: 0.32,
            ease: "power3.in"
        }, 0);
    }
}

if (menuBtn) {
    menuBtn.addEventListener('click', toggleMenu);
}

// Close when clicking outside panel
if (menuContainer) {
    menuContainer.addEventListener('click', (e) => {
        if (isMenuOpen && (e.target === layer1 || e.target === layer2 || e.target === menuContainer)) {
            toggleMenu();
        }
    });
}

// Close when clicking a nav item (and smooth scroll)
menuItems.forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = item.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        
        if (targetSection) {
            toggleMenu();
            setTimeout(() => {
                targetSection.scrollIntoView({ behavior: 'smooth' });
            }, 350); // Wait for close animation
        }
    });
});

// ==========================================
// SCROLL-LINKED CLOUD PARALLAX
// ==========================================
const hero = document.getElementById('hero'); // use correct hero id
const cloudDivider = document.querySelector('.cloud-divider');

if (hero && cloudDivider) {
    let ticking = false;

    function updateCloud() {
        const scrolled = window.scrollY;
        const heroHeight = hero.offsetHeight;
        const windowHeight = window.innerHeight;

        // Progress from 0 (hero top) to 1 (hero bottom)
        const progress = Math.max(0, Math.min(1, 
            scrolled / (heroHeight - windowHeight)
        ));

        // Cloud rises from 100% below to 0% (fully visible) 
        // starts rising at progress 0.3, completes at 0.85
        const start = 0.3;
        const end = 0.85;
        let cloudProgress = 0;
        if (end > start) {
            cloudProgress = Math.max(0, Math.min(1, 
                (progress - start) / (end - start)
            ));
        }

        const translateY = (1 - cloudProgress) * 100;
        cloudDivider.style.transform = `translateY(${translateY}%)`;
        ticking = false;
    }

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(updateCloud);
            ticking = true;
        }
    }, { passive: true });

    updateCloud(); // run on load
}

// ==========================================
// SMOOTH SCROLL
// ==========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(anchor.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

