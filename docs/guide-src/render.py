import asyncio, pathlib
from playwright.async_api import async_playwright

HTML = pathlib.Path("guide.html").resolve()
OUT  = pathlib.Path("reflex-cv-guide.pdf").resolve()

FOOTER = """
<div style="width:100%;font-family:Helvetica,Arial,sans-serif;font-size:7.5pt;
            color:#6b7279;padding:0 16mm;display:flex;justify-content:space-between;">
  <span>reflex-cv &mdash; Code Reading Guide &amp; Field Manual</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto(HTML.as_uri(), wait_until="load")
        await pg.emulate_media(media="print")
        await pg.pdf(path=str(OUT), format="A4", print_background=True,
                     display_header_footer=True,
                     header_template="<div></div>",
                     footer_template=FOOTER,
                     margin={"top":"19mm","bottom":"18mm","left":"16mm","right":"16mm"})
        await b.close()
    print("wrote", OUT)

asyncio.run(main())
