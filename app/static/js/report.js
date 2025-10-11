/* static/js/report.js — minimal frontend that previews nearby helpers,
   submits reports, and polls ETA status. */
(function () {
  const FIND_NEARBY_ENDPOINT = "/find_nearby";
  const CREATE_REPORT_ENDPOINT = "/reports";
  const ETA_STATUS_ENDPOINT = (id) => `/eta_status/${id}`;
  const POLL_INTERVAL = 12 * 1000;
  function el(id){return document.getElementById(id);}
  function create(tag, attrs){const e=document.createElement(tag); if(attrs) Object.keys(attrs).forEach(k=>{ if(k==="text") e.textContent=attrs[k]; else e.setAttribute(k,attrs[k]); }); return e;}
  function fmtMinutesText(s){ if(s==null) return "unknown"; if(s<60) return "<1 min"; const m=Math.round(s/60); return m===1? "1 min": m+" mins"; }

  function setError(msg){ const e=el("report-error"); if(e) e.textContent = msg || ""; console.error(msg||""); }

  async function findNearby(lat, lon, max_radius_km=15, max_results=6){
    try{
      const r = await fetch(FIND_NEARBY_ENDPOINT, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({lat:parseFloat(lat), lon:parseFloat(lon), max_radius_km, max_results}) });
      if(!r.ok){ setError("find_nearby failed: "+r.status); return null; }
      const j = await r.json(); if(j.status && j.status!=="ok"){ setError("find_nearby error: "+JSON.stringify(j)); return null; }
      return j.helpers || [];
    }catch(ex){ setError("find_nearby exception: "+ex); return null; }
  }

  async function submitReport(payload){
    try{
      const r = await fetch(CREATE_REPORT_ENDPOINT, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload) });
      if(!r.ok){ setError("submit failed: "+r.status); return null; }
      const j = await r.json(); if(j.status && j.status!=="ok"){ setError("submit error: "+JSON.stringify(j)); return null; }
      return j;
    }catch(ex){ setError("submit exception: "+ex); return null; }
  }

  function renderHelperCard(h){
    const card = create("div",{class:"helper-card", style:"border:1px solid #ddd;padding:8px;margin:6px 0;border-radius:6px;"});
    const title = create("div",{style:"font-weight:600", text: h.name || ("Helper "+(h.helper_id||h.id||""))});
    card.appendChild(title);
    const meta = create("div",{style:"font-size:0.9em;color:#444", text: `Distance: ${h.dist_km!=null? h.dist_km+" km":"unknown"} • ETA: ${h.eta_text||h.eta||(h.eta_seconds?fmtMinutesText(h.eta_seconds):"unknown")}`});
    card.appendChild(meta);
    const contact = create("div",{style:"margin-top:6px;font-size:0.9em"});
    if(h.phone) contact.innerHTML += `Phone: ${h.phone} `;
    if(h.email) contact.innerHTML += ` Email: ${h.email}`;
    card.appendChild(contact);
    return card;
  }

  function renderLiveHelperRow(h){
    const row = create("div",{class:"live-helper-row", "data-hid": h.helper_id||h.id, style:"border-bottom:1px dashed #eee;padding:8px 0;"});
    const left = create("div",{style:"display:flex;justify-content:space-between;align-items:center"});
    const name = create("div",{style:"font-weight:600", text: h.name || ("Helper "+(h.helper_id||h.id))});
    left.appendChild(name);
    const meta = create("div",{style:"font-size:0.9em;color:#333", text:`ETA: ${h.eta||h.eta_text||(h.eta_seconds?fmtMinutesText(h.eta_seconds):"unknown")}`});
    meta.id = `eta-text-${h.helper_id||h.id}`; left.appendChild(meta);
    row.appendChild(left);
    const sub = create("div",{style:"font-size:0.85em;color:#666;margin-top:6px", text:`Distance: ${h.dist_km!=null? h.dist_km+" km":"unknown"} • Contact: ${h.phone||h.email||"n/a"}`});
    row.appendChild(sub);
    return row;
  }

  let pollTimer = null; let currentReportId = null;

  async function pollEtaAndUpdate(reportId){
    if(!reportId) return;
    try{
      const r = await fetch(ETA_STATUS_ENDPOINT(reportId));
      if(!r.ok){ console.warn("eta_status non-200", r.status); return; }
      const j = await r.json();
      if(j.status && j.status!=="ok"){ console.warn("eta_status payload issue", j); return; }
      const list = j.assigned_helpers || [];
      const liveList = el("live-eta-list"); if(!liveList) return; liveList.innerHTML = "";
      if(!list.length) liveList.appendChild(create("div",{text:"No assigned helpers yet."}));
      else list.forEach(h => liveList.appendChild(renderLiveHelperRow(h)));
    }catch(ex){ console.warn("pollEta error", ex); }
  }

  function startPolling(reportId){
    if(!reportId) return;
    if(pollTimer) clearInterval(pollTimer);
    pollEtaAndUpdate(reportId);
    pollTimer = setInterval(()=> pollEtaAndUpdate(reportId), POLL_INTERVAL);
  }
  function stopPolling(){ if(pollTimer){ clearInterval(pollTimer); pollTimer=null; } }

  function ensureUI(){
    let main = el("report-main");
    if(!main){ main = create("div",{id:"report-main", style:"max-width:900px;margin:10px auto;padding:6px;"}); document.body.appendChild(main); }
    if(!el("report-message")){ const p=create("p"); p.appendChild(create("label",{for:"report-message", text:"Message:"})); p.appendChild(create("br")); p.appendChild(create("textarea",{id:"report-message", rows:"5", cols:"80"})); main.appendChild(p); }
    if(!el("report-lat")){ const latp=create("p"); latp.appendChild(create("label",{for:"report-lat", text:"Latitude:"})); latp.appendChild(create("input",{id:"report-lat", name:"lat", placeholder:"e.g. 12.9716", style:"width:160px;margin-left:8px"})); latp.appendChild(create("label",{for:"report-lon", text:" Longitude:", style:"margin-left:12px"})); latp.appendChild(create("input",{id:"report-lon", name:"lon", placeholder:"e.g. 77.5946", style:"width:160px;margin-left:8px"})); const btnLoc=create("button",{id:"use-my-location", type:"button", text:"Use my location"}); btnLoc.style.marginLeft="12px"; latp.appendChild(btnLoc); main.appendChild(latp); }
    if(!el("btn-preview-nearby")){ const previewP=create("p"); previewP.appendChild(create("button",{id:"btn-preview-nearby", type:"button", text:"Preview nearby stations"})); previewP.appendChild(create("button",{id:"btn-submit-report", type:"button", text:"Submit report", style:"margin-left:12px"})); main.appendChild(previewP); }
    if(!el("nearby-results")){ const wrap=create("div",{id:"nearby-results", style:"margin-top:12px;"}); wrap.appendChild(create("h3",{text:"Nearby Stations / Helpers"})); wrap.appendChild(create("div",{id:"nearby-list"})); main.appendChild(wrap); }
    if(!el("live-eta-panel")){ const panel=create("div",{id:"live-eta-panel", style:"margin-top:16px;border-top:'1px solid #eee';padding-top:10px;"}); panel.appendChild(create("h3",{text:"Live ETA updates"})); panel.appendChild(create("div",{id:"live-eta-list"})); main.appendChild(panel); }
    if(!el("report-error")){ const err=create("div",{id:"report-error", style:"color:#a00;margin-top:10px;"}); main.appendChild(err); }
  }

  function wire(){
    ensureUI();
    const btnPreview = el("btn-preview-nearby");
    const btnSubmit = el("btn-submit-report");
    const useMeBtn = el("use-my-location");
    btnPreview && btnPreview.addEventListener("click", async function(){
      const lat = el("report-lat").value.trim(); const lon = el("report-lon").value.trim();
      if(!lat||!lon){ setError("Please enter latitude and longitude (or use 'Use my location')."); return; }
      const helpers = await findNearby(lat, lon);
      const list = el("nearby-list"); if(!list) return; list.innerHTML="";
      if(!helpers || !helpers.length) list.appendChild(create("div",{text:"No helpers found nearby."}));
      else helpers.forEach(h=> list.appendChild(renderHelperCard(h)));
    });
    btnSubmit && btnSubmit.addEventListener("click", async function(){
      setError(""); stopPolling();
      const msg = el("report-message").value.trim(); const lat = el("report-lat").value.trim(); const lon = el("report-lon").value.trim();
      if(!msg){ setError("Please enter a message."); return; }
      const payload = { message: msg }; if(lat&&lon){ payload.lat=parseFloat(lat); payload.lon=parseFloat(lon); }
      const resp = await submitReport(payload); if(!resp) return;
      const liveList = el("live-eta-list"); liveList.innerHTML="";
      if(resp.assigned_helpers && resp.assigned_helpers.length) resp.assigned_helpers.forEach(h=> liveList.appendChild(renderLiveHelperRow(h)));
      else liveList.appendChild(create("div",{text:"Report submitted. Waiting for helper assignment..."}));
      if(resp.report_id){ startPolling(resp.report_id); const panel = el("live-eta-panel"); if(panel) panel.scrollIntoView({behavior:"smooth"}); }
    });
    useMeBtn && useMeBtn.addEventListener("click", function(){
      if(!navigator.geolocation){ setError("Geolocation not supported."); return; }
      setError("Detecting your location...");
      navigator.geolocation.getCurrentPosition(function(pos){ setError(""); el("report-lat").value = pos.coords.latitude.toFixed(6); el("report-lon").value = pos.coords.longitude.toFixed(6); }, function(err){ setError("Geolocation error: " + (err.message||err.code)); }, { timeout:10000, enableHighAccuracy:true, maximumAge:10000 });
    });
  }

  if(document.readyState === "loading"){ document.addEventListener("DOMContentLoaded", wire); } else { wire(); }
  window.__disaster_report_ui = { startPolling, stopPolling, findNearby, submitReport };
})();
