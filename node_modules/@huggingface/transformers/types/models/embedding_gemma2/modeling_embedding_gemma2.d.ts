export class EmbeddingGemma2PreTrainedModel extends PreTrainedModel {
}
/**
 * The EmbeddingGemma 2 model, which embeds any mix of text, images, audio and video into a single vector.
 *
 * The vision and audio encoders turn media into soft tokens, and the text model places them at their placeholder
 * tokens, runs the bidirectional text encoder, and returns the per-token `last_hidden_state` together with the
 * mean-pooled, L2-normalized `sentence_embedding`.
 *
 * The encoders are only loaded when the config has a `vision_config` / `audio_config`. For text-only use, remove them:
 * ```javascript
 * const model_id = 'onnx-community/embeddinggemma-2-ONNX';
 * const config = await AutoConfig.from_pretrained(model_id);
 * config.vision_config = config.audio_config = null;
 * const model = await AutoModel.from_pretrained(model_id, { config });
 * ```
 */
export class EmbeddingGemma2Model extends EmbeddingGemma2PreTrainedModel {
    forward({ input_ids, attention_mask, pixel_values, image_position_ids, pixel_values_videos, video_position_ids, input_features, input_features_mask, }: {
        input_ids: any;
        attention_mask: any;
        pixel_values?: any;
        image_position_ids?: any;
        pixel_values_videos?: any;
        video_position_ids?: any;
        input_features?: any;
        input_features_mask?: any;
    }): Promise<any>;
}
import { PreTrainedModel } from '../modeling_utils.js';
//# sourceMappingURL=modeling_embedding_gemma2.d.ts.map