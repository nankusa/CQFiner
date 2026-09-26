"""Decode learned DETR mask queries using the historical class/mask scoring."""


class DetrInferenceMixin:
    def _predict_detr_site_outputs(self, batch, dataset_name, out):
        logits = out["site_logits"]
        masks = out["site_mask_logits"]
        valid = out["site_query_mask"]
        sample_ids = out["site_sample_ids"]
        if logits.ndim != 3 or masks.ndim != 3 or logits.shape[:2] != masks.shape[:2]:
            raise ValueError("DETR class and mask outputs must share [B, Q] dimensions.")
        if valid.shape != logits.shape[:2] or sample_ids.shape != (logits.size(0),):
            raise ValueError("DETR padding masks or sample IDs have invalid shapes.")
        if self.model.site_detector.query_source != "learned":
            raise ValueError("DETR mask decoding requires learned object queries.")
        names = self._batch_sample_ids(batch)
        outputs = []
        for index, sample_id in enumerate(sample_ids.tolist()):
            selected = batch.batch == sample_id
            count = int(selected.sum())
            if count == 0 or masks.size(-1) < count:
                raise ValueError("DETR residue mask does not cover the protein.")
            prediction = self._build_site_detector_prediction(
                host_pos=batch.pos[selected],
                site_logits=logits[index, valid[index]],
                site_mask_logits=masks[index, valid[index], :count],
                mask_threshold=self.site_mask_threshold,
            )
            outputs.append({
                "dataset_name": dataset_name,
                "sample_idx": sample_id,
                "sample_id": names[sample_id],
                "atom_residue_indices": None,
                "prediction": prediction,
            })
        if not outputs:
            raise ValueError("DETR received an empty protein batch.")
        self._test_site_prediction_outputs[dataset_name].extend(outputs)
        return outputs
