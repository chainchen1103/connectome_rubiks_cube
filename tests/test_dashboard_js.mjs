// Run with node --test tests/test_dashboard_js.mjs. No browser dependencies.
// These assertions protect the 3-D turn animation's geometry, independently
// of the Python environment's sticker permutation tests.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('../connectome_lab/assets/dashboard.html', import.meta.url), 'utf8');
const source = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1].replace(/\ninitialize\(\);\n/, '\n');
const payload = {metadata:{},anatomy:{positions:[],ids:[]},circuit:{ids:[],positions:[]},cube:{frames:[]},fly:{frames:[]}};
const context = {window:{matchMedia:()=>({matches:false})},document:{getElementById:()=>({textContent:JSON.stringify(payload)})}};
vm.runInNewContext(source, context);
const {rotateAxis,moveSpec,cubeStickerVertices,groupEpisodes,normalizePositions,normalizeSkeletonSegments,rewardDisplay} = context.window.ConnectomeDashboard;
const plain = object => JSON.parse(JSON.stringify(object));
const near = (actual, expected) => actual.forEach((v,i) => assert.ok(Math.abs(v-expected[i])<1e-9, `${actual} != ${expected}`));

test('U, R, F move directions agree with the Python geometric convention', () => {
  const source = [2,2,2];
  for (const [action,expected] of [[0,[-2,2,2]],[2,[2,2,-2]],[4,[2,-2,2]]]) {
    const {axis,quarter} = moveSpec(action);
    near(rotateAxis(source,axis,quarter*Math.PI/2), expected);
  }
});

test('all animated quarter-turn endpoints form exact, bijective sticker permutations', () => {
  for (const size of [2,3]) {
    const extent = size-1, values = Array.from({length:size},(_,i)=>-extent+i*2);
    const geometry=[];
    for(const [axis,sign] of [[1,1],[0,1],[2,1],[1,-1],[0,-1],[2,-1]]) {
      const varying=[0,1,2].filter(a=>a!==axis);
      for(const a of values)for(const b of values){const position=[0,0,0],normal=[0,0,0];position[axis]=sign*extent;position[varying[0]]=a;position[varying[1]]=b;normal[axis]=sign;geometry.push({position,normal});}
    }
    const key=(p,n)=>[...p,...n].map(v=>Math.round(v)).join(',');
    const valid=new Set(geometry.map(g=>key(g.position,g.normal)));
    for(let action=0;action<12;action++) {
      const destinations=new Set();
      for(const sticker of geometry) {
        const endpoint=cubeStickerVertices(sticker,size,action,1);
        const center=[0,1,2].map(axis=>endpoint.vertices.reduce((sum,v)=>sum+v[axis],0)/4-endpoint.normal[axis]*.965);
        const destination=key(center,endpoint.normal);
        assert.ok(valid.has(destination), `invalid size ${size}, action ${action}: ${destination}`);
        destinations.add(destination);
        const zero=cubeStickerVertices(sticker,size,action,0);
        near(zero.normal,sticker.normal);
        const spec=moveSpec(action);
        if(sticker.position[spec.axis]!==spec.sign*extent)near(endpoint.normal,sticker.normal);
      }
      assert.equal(destinations.size,geometry.length,`non-bijective action ${action}`);
    }
  }
});

test('inverse moves and four turns preserve geometric vectors', () => {
  for(let action=0;action<12;action++) {
    const spec=moveSpec(action),inverse=moveSpec(action^1),v=[2,-1,3];
    near(rotateAxis(rotateAxis(v,spec.axis,spec.quarter*Math.PI/2),inverse.axis,inverse.quarter*Math.PI/2),v);
    let p=v;for(let i=0;i<4;i++)p=rotateAxis(p,spec.axis,spec.quarter*Math.PI/2);near(p,v);
  }
});

test('anatomical normalization preserves shared metric scale across all axes', () => {
  assert.deepEqual(plain(normalizePositions([[10,20,30],[30,100,50]])),{center:[20,60,40],extent:80});
  assert.deepEqual(plain(normalizePositions([])),{center:[0,0,0],extent:1});
});

test('source skeleton segments preserve adjacency and the anatomical coordinate transform', () => {
  const skeletons={neurons:[{segments:[[10,20,30,30,100,50],[100,200,300,101,201,301]]},{segments:[[1,2,3], [0,0,0,NaN,1,2]]}]};
  const segments=normalizeSkeletonSegments(skeletons,{center:[20,60,40],extent:80});
  assert.equal(segments.length,2);
  near(segments[0][0],[-10,-40,-10].map(value=>value*1.74/80));
  near(segments[0][1],[10,40,10].map(value=>value*1.74/80));
  near(segments[1][0],[80,140,260].map(value=>value*1.74/80));
  assert.deepEqual(plain(normalizeSkeletonSegments(undefined,{center:[0,0,0],extent:1})),[]);
});

test('replay grouping never interpolates between independent reset episodes', () => {
  assert.deepEqual(plain(groupEpisodes([{episode:1},{episode:1},{episode:2},{episode:3},{episode:3}])),[{id:1,start:0,end:1},{id:2,start:2,end:2},{id:3,start:3,end:4}]);
});

test('training reward display retains small penalties and distinguishes missing data from zero', () => {
  const result=rewardDisplay({potential_before:.5,potential_after:.5,progress_reward:0,step_penalty:-.002,inverse_penalty:-.01,revisit_penalty:-.02,solved_bonus:0,reward:-.032});
  assert.equal(result.step,'-0.0020');
  assert.equal(result.total,'-0.0320');
  assert.equal(result.progress,'0.0000');
  assert.equal(result.potential,'0.500 → 0.500');
  assert.equal(rewardDisplay(undefined).total,'—');
  assert.equal(rewardDisplay({reward:0}).total,'0.0000');
});
