#!/usr/bin/env node

/**
 * HTML to PNG Converter using Puppeteer
 * Converts standalone HTML visualizations to high-quality PNG images
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function convertHtmlToPng(htmlFilePath, outputPath, options = {}) {
    const {
        width = 1200,
        height = 800,
        deviceScaleFactor = 2,
        waitTime = 3000,
        selector = '.container',
        quality = 100
    } = options;

    console.log(`Converting ${htmlFilePath} to PNG...`);

    try {
        // Launch browser
        const browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();

        // Set viewport with high DPI
        await page.setViewport({
            width: width,
            height: height,
            deviceScaleFactor: deviceScaleFactor
        });

        // Load HTML file
        const htmlPath = path.resolve(htmlFilePath);
        const htmlUrl = `file://${htmlPath}`;
        
        console.log(`Loading: ${htmlUrl}`);
        
        // Navigate to HTML file with network idle wait
        await page.goto(htmlUrl, { 
            waitUntil: 'networkidle0',
            timeout: 30000
        });

        // Wait for D3.js rendering to complete
        console.log(`Waiting ${waitTime}ms for rendering...`);
        await page.waitForTimeout(waitTime);

        // Check if page loaded correctly
        const title = await page.title();
        console.log(`Page title: ${title}`);

        // Wait for specific selector if provided
        if (selector) {
            try {
                await page.waitForSelector(selector, { timeout: 10000 });
                console.log(`Found selector: ${selector}`);
            } catch (error) {
                console.warn(`Selector ${selector} not found, proceeding anyway`);
            }
        }

        // Log any console errors from the page
        page.on('console', msg => {
            if (msg.type() === 'error') {
                console.error('Page error:', msg.text());
            } else if (msg.type() === 'log') {
                console.log('Page log:', msg.text());
            }
        });

        // Take screenshot
        const screenshotOptions = {
            path: outputPath,
            type: 'png',
            fullPage: false
        };

        if (selector) {
            // Screenshot specific element
            const element = await page.$(selector);
            if (element) {
                await element.screenshot(screenshotOptions);
                console.log(`Screenshot of ${selector} saved to: ${outputPath}`);
            } else {
                // Fallback to full page
                await page.screenshot({...screenshotOptions, fullPage: true});
                console.log(`Element not found, full page screenshot saved to: ${outputPath}`);
            }
        } else {
            // Full page screenshot
            await page.screenshot({...screenshotOptions, fullPage: true});
            console.log(`Full page screenshot saved to: ${outputPath}`);
        }

        await browser.close();

        // Verify file was created
        if (fs.existsSync(outputPath)) {
            const stats = fs.statSync(outputPath);
            console.log(`✓ PNG created successfully (${(stats.size / 1024).toFixed(1)} KB)`);
            return true;
        } else {
            console.error('✗ PNG file was not created');
            return false;
        }

    } catch (error) {
        console.error('Error converting HTML to PNG:', error);
        return false;
    }
}

async function convertAllVisualizations() {
    const visualizationDir = path.join(__dirname);
    const outputDir = path.join(__dirname, '../../results/figures');

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const htmlFiles = [
        {
            input: 'ml_dili_prediction_results.html',
            output: 'ml_dili_prediction_analysis.png'
        },
        {
            input: 'pk_oxygen_correlation_analysis.html',
            output: 'pk_oxygen_final_summary.png'
        },
        {
            input: 'data_quality_verification.html',
            output: 'data_quality_verification.png'
        }
    ];

    const results = [];

    for (const file of htmlFiles) {
        const inputPath = path.join(visualizationDir, file.input);
        const outputPath = path.join(outputDir, file.output);

        if (!fs.existsSync(inputPath)) {
            console.error(`HTML file not found: ${inputPath}`);
            continue;
        }

        console.log(`\n${'='.repeat(60)}`);
        console.log(`Converting: ${file.input} → ${file.output}`);
        console.log('='.repeat(60));

        const success = await convertHtmlToPng(inputPath, outputPath, {
            width: 1200,
            height: 800,
            deviceScaleFactor: 2,
            waitTime: 4000,
            selector: '.container'
        });

        results.push({ file: file.input, success });
    }

    // Summary
    console.log(`\n${'='.repeat(60)}`);
    console.log('CONVERSION SUMMARY');
    console.log('='.repeat(60));

    results.forEach(result => {
        const status = result.success ? '✓' : '✗';
        console.log(`${status} ${result.file}`);
    });

    const successCount = results.filter(r => r.success).length;
    console.log(`\nCompleted: ${successCount}/${results.length} files converted successfully`);
}

// Command line usage
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        // Convert all visualizations
        convertAllVisualizations().catch(console.error);
    } else if (args.length >= 2) {
        // Convert specific file
        const [inputFile, outputFile] = args;
        const options = {};

        // Parse additional options
        for (let i = 2; i < args.length; i += 2) {
            const key = args[i];
            const value = args[i + 1];

            if (key === '--width') options.width = parseInt(value);
            else if (key === '--height') options.height = parseInt(value);
            else if (key === '--scale') options.deviceScaleFactor = parseFloat(value);
            else if (key === '--wait') options.waitTime = parseInt(value);
            else if (key === '--selector') options.selector = value;
        }

        convertHtmlToPng(inputFile, outputFile, options).catch(console.error);
    } else {
        console.log('Usage:');
        console.log('  node html_to_png.js                           # Convert all visualizations');
        console.log('  node html_to_png.js input.html output.png     # Convert specific file');
        console.log('  node html_to_png.js input.html output.png --width 1600 --height 1000 --scale 2');
    }
}

module.exports = { convertHtmlToPng, convertAllVisualizations };