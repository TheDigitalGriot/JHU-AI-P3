import * as THREE from "three/webgpu";

class WebGPUContext {
	constructor(container) {
		if (!!WebGPUContext.instance) {
			return WebGPUContext.instance;
		}

		this.container = container;
		this.renderer = null;
		this.canvas = null;
		this.pixelRatio = Math.min(window.devicePixelRatio, 2.0);

		WebGPUContext.instance = this;
	}

	async init() {
		this.canvas = this.#createCanvas();
		this.renderer = new THREE.WebGPURenderer({
			canvas: this.canvas,
			antialias: false,
		});

		await this.renderer.init();

		const { width, height } = this.getFullScreenDimensions();
		this.renderer.setSize(width, height);
		this.renderer.setPixelRatio(this.pixelRatio);
		this.renderer.shadowMap.enabled = false;
		this.renderer.autoClear = false;
		this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
	}

	getFullScreenDimensions() {
		const el = document.createElement("div");
		Object.assign(el.style, {
			height: "100lvh",
			width: "100lvw",
			position: "absolute",
			visibility: "hidden",
		});
		document.body.appendChild(el);
		const width = el.offsetWidth;
		const height = el.offsetHeight;
		document.body.removeChild(el);
		return { width, height };
	}

	#createCanvas() {
		const canvas = document.createElement("canvas");
		canvas.id = "btd-hero-canvas";
		canvas.style.position = "fixed";
		canvas.style.left = 0;
		canvas.style.top = 0;
		canvas.style.zIndex = 1;
		canvas.style.pointerEvents = "none";
		document.body.appendChild(canvas);
		return canvas;
	}

	onResize(width, height) {
		this.pixelRatio = Math.min(window.devicePixelRatio, 2);
		this.renderer.setSize(width, height);
		this.renderer.setPixelRatio(this.pixelRatio);
	}

	dispose() {
		try { this.renderer?.dispose?.(); } catch {}
		if (this.canvas?.parentNode) this.canvas.parentNode.removeChild(this.canvas);
		WebGPUContext.instance = null;
	}
}

export default WebGPUContext;
