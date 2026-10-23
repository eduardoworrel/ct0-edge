declare const EmbeddingGemma2VideoProcessor_base: new () => {
    (...args: any[]): any;
    _call(...args: any[]): any;
};
/**
 * Processes videos as sequences of frames, each going through the Gemma 4 image processor with the video budget of
 * soft tokens (`max_soft_tokens`). Frames are used as decoded (e.g., `load_video(url, { fps: 1 })`), and videos with
 * more than `max_frames` frames are subsampled according to `overflow_strategy` ('uniform' or 'truncate').
 */
export class EmbeddingGemma2VideoProcessor extends EmbeddingGemma2VideoProcessor_base {
    /** @param {Record<string, any>} config */
    constructor(config: Record<string, any>);
    config: Record<string, any>;
    frame_processor: Gemma4ImageProcessor;
    max_frames: any;
    overflow_strategy: any;
    /**
     * @template T
     * @param {T[]} frames
     * @returns {T[]} At most `max_frames` of the frames.
     */
    sample_frames<T>(frames: T[]): T[];
    /**
     * @param {import('../../utils/video.js').RawVideo|import('../../utils/video.js').RawVideo[]} videos
     */
    _call(videos: import("../../utils/video.js").RawVideo | import("../../utils/video.js").RawVideo[]): Promise<{
        pixel_values_videos: any;
        video_position_ids: any;
        num_frames_per_video: number[];
        num_soft_tokens_per_video: any[];
    }>;
}
import { Gemma4ImageProcessor } from '../gemma4/image_processing_gemma4.js';
export {};
//# sourceMappingURL=video_processing_embedding_gemma2.d.ts.map