export class EmbeddingGemma2Processor extends Gemma4Processor {
    video_token: any;
    video_processor: EmbeddingGemma2VideoProcessor;
    /**
     * Prepares any mix of text, images, audio and videos for `EmbeddingGemma2Model`.
     *
     * Each `<|image|>`, `<|audio|>` or `<|video|>` placeholder of the text is expanded to the soft tokens of the next
     * item of that modality, in order across the batch. Without text, each sample holds the placeholders of its media
     * (images, then videos, then audio). As in Python, a flat list of images is then a single sample, and a flat list
     * of audio clips or videos is one sample per item: pass one list per sample to group them differently.
     *
     * @param {string|string[]|null} text The text, with a placeholder per media item.
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]|import('../../utils/image.js').RawImage[][]} [images]
     * @param {Float32Array|Float32Array[]|Float32Array[][]} [audio] Mono audio at the feature extractor's sampling rate (16 kHz).
     * @param {import('../../utils/video.js').RawVideo|import('../../utils/video.js').RawVideo[]|import('../../utils/video.js').RawVideo[][]} [videos]
     * @param {Object} [options] Additional options for the tokenizer.
     */
    _call(text: string | string[] | null, images?: import("../../utils/image.js").RawImage | import("../../utils/image.js").RawImage[] | import("../../utils/image.js").RawImage[][], audio?: Float32Array | Float32Array[] | Float32Array[][], videos?: import("../../utils/video.js").RawVideo | import("../../utils/video.js").RawVideo[] | import("../../utils/video.js").RawVideo[][], options?: any): Promise<any>;
    /**
     * Extracts the features of each clip and pads them to the longest one.
     * @param {Float32Array[]} clips
     */
    _process_audio(clips: Float32Array[]): Promise<{
        input_features: Tensor;
        input_features_mask: Tensor;
        num_soft_tokens_per_audio: number[];
    }>;
}
import { Gemma4Processor } from '../gemma4/processing_gemma4.js';
import { EmbeddingGemma2VideoProcessor } from './video_processing_embedding_gemma2.js';
import { Tensor } from '../../utils/tensor.js';
//# sourceMappingURL=processing_embedding_gemma2.d.ts.map