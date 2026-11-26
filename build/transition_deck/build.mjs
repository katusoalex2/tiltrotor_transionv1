import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { slides } from "./content.mjs";

const root = process.cwd();
const runtime = path.join(process.env.USERPROFILE, ".cache/codex-runtimes/codex-primary-runtime/dependencies");
const skill = path.join(process.env.USERPROFILE, ".codex/plugins/cache/openai-primary-runtime/presentations/26.1119.11507/skills/presentations");
const packageUrl = pathToFileURL(path.join(runtime, "node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"));
const { Presentation, PresentationFile } = await import(packageUrl.href);
const stage = path.join(root, "build/transition_deck");
const output = path.join(stage, "revised/Forward-Only Tiltrotor Transition - Engineering Deep Dive.pptx");
const canvas = {width:1280,height:720};
const font = "Arial";
const navy = "#142C42";
const body = "#253E50";
const secondary = "#486376";
const teal = "#147D72";
const green = "#A5E2D3";
const paper = "#F5F8FA";
const white = "#FFFFFF";
const pres = Presentation.create({slideSize:canvas});
const slideTables = [];
const slideCharts = [];

function text(s, value, left, top, width, height, opts={}) {
  const shape = s.shapes.add({name:opts.name || value.slice(0,60),geometry:"textbox",
    position:{left,top,width,height},fill:"none",line:{fill:"none",width:0}});
  shape.text = value;
  shape.text.style = {
    typeface:font, fontSize:opts.size || 25, bold:opts.bold || false, color:opts.color || body,
    insets:0, alignment:opts.align || "left", verticalAlignment:"top", autoFit:"none",
  };
  return shape;
}

function noteLine(s, d, y=607) {
  if (d.close) text(s,d.close,72,y,1138,58,{size:22,color:teal,bold:true});
}

function footer(s,d,index) {
  let source = d.source.replaceAll("figures/transition_study/refined_corridor/","refined_corridor/");
  if (source.length > 115) source = source.slice(0,110) + "…";
  text(s,source,72,682,1085,23,{size:14,color:secondary});
  text(s,String(index).padStart(2,"0"),1174,680,34,25,{size:16,color:secondary,align:"right"});
}

function header(s,d,index) {
  s.background.fill=paper;
  text(s,d.section.toUpperCase(),72,37,1138,25,{size:16,bold:true,color:teal});
  text(s,d.title,72,75,1140,68,{size:d.title.length > 54 ? 42 : 45,bold:true,color:navy});
  footer(s,d,index);
}

async function picture(s,filename,box) {
  const file = await fs.readFile(path.join(root,filename));
  s.images.add({blob:file.buffer.slice(file.byteOffset,file.byteOffset+file.byteLength),contentType:"image/png",
    alt:path.basename(filename),fit:"contain",position:box});
}

for (const [i,d] of slides.entries()) {
  const s = pres.slides.add(canvas);
  s.speakerNotes.text = d.notes + "\n\nProject reference: " + d.source;
  if (d.kind === "cover") {
    s.background.fill=navy;
    text(s,d.section,80,91,1110,32,{size:18,color:green,bold:true});
    text(s,d.title,80,157,1120,208,{size:77,bold:true,color:white});
    text(s,d.sub,83,412,1096,80,{size:33,color:white});
    text(s,d.detail,83,582,1100,52,{size:24,color:green});
    continue;
  }
  header(s,d,i+1);
  if (d.kind === "metrics") {
    text(s,d.intro,72,168,1100,80,{size:29});
    d.metrics.forEach(([value,label],j) => {
      const left = 72 + 288*j;
      text(s,value,left,294,278,85,{size:57,bold:true,color:navy});
      text(s,label,left,389,260,63,{size:22,color:secondary});
    });
    noteLine(s,d,550);
  } else if (d.kind === "figure") {
    await picture(s,d.image,{left:72,top:161,width:790,height:485});
    d.points.forEach((p,j) => {
      text(s,"•",897,177+j*108,19,37,{size:24,color:teal,bold:true});
      text(s,p,925,177+j*108,279,102,{size:23});
    });
  } else if (d.kind === "formula") {
    d.formulas.forEach(([label,value],j) => {
      const top = 166+j*134;
      text(s,label,72,top,1138,32,{size:22,bold:true,color:teal});
      text(s,value,72,top+39,1138,62,{size:38,bold:true,color:navy});
    });
    noteLine(s,d,590);
  } else if (d.kind === "table") {
    slideTables.push(i+1);
    const rowCount = d.values.length;
    const rowH = rowCount >= 8 ? 54 : rowCount === 7 ? 60 : 70;
    const tb = s.tables.add({rows:rowCount,columns:d.widths.length,values:d.values,
      columnWidths:d.widths,left:72,top:160,width:1138,height:rowCount*rowH});
    tb.styleOptions = {headerRow:false,bandedRows:false,firstColumn:false};
    for (let row=0;row<rowCount;row++) {
      tb.rows[row].height=rowH;
      for (let col=0;col<d.widths.length;col++) {
        const cell = tb.getCell(row,col);
        cell.fill = row === 0 ? navy : row%2 ? white : "#EAF1F5";
        cell.text.style = {typeface:font,fontSize:row === 0 ? 22 : 21,
          bold:row === 0 || col === 0,color:row === 0 ? white : body,
          verticalAlignment:"middle",insets:{left:12,right:9,top:5,bottom:5}};
      }
    }
    tb.borders.assign({fill:"#D6E1E7",width:1,style:"solid"});
    noteLine(s,d,Math.max(550,196+rowCount*rowH));
  } else if (d.kind === "chart") {
    slideCharts.push(i+1);
    const axisStyle={typeface:font,fill:body,fontSize:19};
    s.charts.add("bar", {
      position:{left:162,top:158,width:960,height:390},
      categories:d.categories,
      series:d.series.map(([name,values,fill])=>({name,values,fill})),
      barOptions:{direction:"column",grouping:"stacked",gapWidth:100,overlap:100},
      hasLegend:true,legend:{position:"bottom",overlay:false,textStyle:axisStyle},
      xAxis:{textStyle:axisStyle},
      yAxis:{min:0,max:30,majorUnit:5,textStyle:axisStyle,
        title:{text:"Number of common cases",textStyle:axisStyle},
        majorGridlines:{fill:"#D6E1E7",width:1,style:"solid"}},
      dataLabels:{showValue:true,position:"center",textStyle:{...axisStyle,fontSize:20,bold:true}},
      chartFill:paper,plotFill:paper
    });
    noteLine(s,d,583);
  } else {
    const pitch = d.items.length === 5 ? 102 : 119;
    d.items.forEach(([label,explanation],j) => {
      const y = 166 + pitch*j;
      text(s,label,72,y,288,88,{size:24,bold:true,color:teal});
      text(s,explanation,370,y,838,88,{size:25});
    });
  }
}

const candidate = path.join(stage,"candidate.pptx");
await fs.mkdir(path.dirname(output),{recursive:true});
await (await PresentationFile.exportPptx(pres)).save(candidate);
console.log("Draft slides:",pres.slides.items.length);

const native = path.join(runtime,"node/node_modules");
process.env.PATH = path.join(runtime,"bin") + path.delimiter + process.env.PATH;
process.env.NODE_PATH = native;
process.env.RUNTIME_NODE_MODULES = native;
const toolsDir = path.join(skill,"container_tools");
const { finalizePresentation } = await import(pathToFileURL(path.join(toolsDir,"artifact_tool_utils.mjs")).href);
const args = {
  workspaceDir:root,candidatePath:candidate,finalPath:output,
  pythonExecutable:path.join(runtime,"python/python.exe"),
  integrityValidatorPath:path.join(toolsDir,"inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(toolsDir,"inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit",
    ...slideTables.flatMap(number=>["--require-native-table-slide",String(number)])],
  explicitTotalSlideCount:slides.length,
  requiredNativeTableOwnerSlides:slideTables,requiredNativeChartOwnerSlides:slideCharts,
  materializeLiteralChartWorkbooks:true,fontPolicy:{basis:"design",families:[font]},
  verifyArtifactToolImport:true,receiptPath:path.join(stage,"validation-revised.json"),
};
await finalizePresentation(args);
console.log(output);
