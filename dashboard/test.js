const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    // Log browser console messages
    page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
    page.on('pageerror', err => console.log('BROWSER ERROR:', err.toString()));
    
    await page.goto('file://c:/Users/adity/Desktop/ISRO-Hackathon-ka-kaam/dashboard/index.html', { waitUntil: 'networkidle0' });
    
    await page.setViewport({ width: 800, height: 800 });
    
    console.log('Clicking menu btn...');
    await page.click('#mobile-menu-btn');
    
    await page.waitForTimeout(2000);
    
    // Check if menu-text is translated to 0
    const textTransforms = await page.evaluate(() => {
        const els = document.querySelectorAll('.menu-text');
        return Array.from(els).map(el => el.style.transform);
    });
    console.log('Transforms:', textTransforms);
    
    await browser.close();
})();
