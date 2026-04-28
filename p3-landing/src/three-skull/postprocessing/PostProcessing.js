import * as THREE from "three/webgpu";
import {
	pass,
	vec3,
	screenUV,
	time,
	float,
	sub,
	sin,
	mul,
	add,
	mix,
	dot,
	clamp,
	Fn,
} from "three/tsl";
import { mx_noise_float } from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";

export default class PostProcessing {
	constructor(renderer, solidScene, wireScene, camera, fluidMaskNode) {
		this.pipeline = new THREE.RenderPipeline(renderer);
		this.solidScene = solidScene;
		this.wireScene = wireScene;
		this.camera = camera;
		this.fluidMaskNode = fluidMaskNode;

		this.#compose();
	}

	#compose() {
		// Render both scenes
		const solidPass = pass(this.solidScene, this.camera);
		const solidColor = solidPass.getTextureNode("output");

		const wirePass = pass(this.wireScene, this.camera);
		const wireColor = wirePass.getTextureNode("output");

		// Bloom on solid scene
		const bloomPass = bloom(solidColor.sample(screenUV), 0.4, 0.05);

		// Scan lines on bloom only (darken only)
		const scanRaw = sin(mul(screenUV.y, float(1250.0)));
		const scanDarken = clamp(scanRaw, -1.0, 0.0).mul(-0.15);
		const scanLines = sub(float(1.0), scanDarken);
		const bloomWithScanLines = bloomPass.mul(scanLines);

		// Fluid mask composites solid ↔ wire
		const fluidMask = sub(float(1.0), this.fluidMaskNode.sample(screenUV).r);
		const blended = mix(
			bloomWithScanLines,
			wireColor.sample(screenUV),
			fluidMask,
		);

		// Film grain
		const noise = mx_noise_float(
			vec3(screenUV.mul(2000.0), time.mul(20.0)),
		).mul(0.015);

		// Combine effects
		const withEffects = blended.sub(noise);

		// Slight desaturation
		const luminance = dot(withEffects, vec3(0.299, 0.587, 0.114));
		const desaturated = mix(
			vec3(luminance, luminance, luminance),
			withEffects,
			float(0.985),
		);

		const lowContrast = mix(vec3(0.0, 0.0, 0.2), desaturated, float(0.9));

		this.pipeline.outputNode = lowContrast;
	}

	render() {
		this.pipeline.render();
	}

	dispose() {
		this.pipeline.dispose();
	}
}
