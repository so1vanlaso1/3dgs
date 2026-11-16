"""Review lowest sharpness-score candidates beside a textured control."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

root=Path('results/school_dataset_audit_20261109')
canvas=Image.new('RGB',(1280,880),'white')
draw=ImageDraw.Draw(canvas)
for row,scene in enumerate(('room','counter','kitchen','bonsai')):
    data=json.loads((root/(scene+'_factor2.json')).read_text())
    ranked=sorted(data['rows'],key=lambda r:r['laplacian_variance'])
    for col,rec in enumerate(ranked[:3]+ranked[-1:]):
        image=Image.open(Path(data['data'])/'images'/rec['name'])
        image.thumbnail((312,185))
        x,y=col*320,row*220
        canvas.paste(image,(x,y+30))
        draw.text((x+3,y+3),scene+' '+rec['name']+' score='+str(round(rec['laplacian_variance'])),fill='black')
canvas.save(root/'blur_candidate_contact_sheet.jpg')
