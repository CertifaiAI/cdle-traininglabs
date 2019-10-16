package global.skymind.training.object_detection.dataHelpers;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.api.util.files.URIUtil;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.datavec.image.data.Image;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.util.ImageUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.concurrency.AffinityManager.Location;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ObjectDetectionRRCustom extends BaseImageRecordReader{
    private final int gridW;
    private final int gridH;
    private final ImageObjectLabelProvider labelProvider;
    protected Image currentImage;

    public ObjectDetectionRRCustom(int height, int width, int channels, int gridH, int gridW, ImageObjectLabelProvider labelProvider) {
        super((long)height, (long)width, (long)channels, (PathLabelGenerator)null, (ImageTransform)null);
        this.gridW = gridW;
        this.gridH = gridH;
        this.labelProvider = labelProvider;
        this.appendLabel = labelProvider != null;
    }

    public ObjectDetectionRRCustom(int height, int width, int channels, int gridH, int gridW, ImageObjectLabelProvider labelProvider, ImageTransform imageTransform) {
        super((long)height, (long)width, (long)channels, (PathLabelGenerator)null, (ImageTransform)null);
        this.gridW = gridW;
        this.gridH = gridH;
        this.labelProvider = labelProvider;
        this.appendLabel = labelProvider != null;
        this.imageTransform = imageTransform;
    }

    public List<Writable> next() {
        return (List)this.next(1).get(0);
    }

    public void initialize(InputSplit split) throws IOException {
        if (this.imageLoader == null) {
            this.imageLoader = new NativeImageLoader(this.height, this.width, this.channels, this.imageTransform);
        }

        this.inputSplit = split;
        URI[] locations = split.locations();
        Set<String> labelSet = new HashSet();
        if (locations != null && locations.length >= 1) {
            URI[] var4 = locations;
            int var5 = locations.length;

            for(int var6 = 0; var6 < var5; ++var6) {
                URI location = var4[var6];
                List<ImageObject> imageObjects = this.labelProvider.getImageObjectsForPath(location);
                Iterator var9 = imageObjects.iterator();

                while(var9.hasNext()) {
                    ImageObject io = (ImageObject)var9.next();
                    String name = io.getLabel();
                    if (!labelSet.contains(name)) {
                        labelSet.add(name);
                    }
//                    Check list of labels
//                    System.out.println(labelSet);

//                    System.out.println("Label: "+io.getLabel());
//                    System.out.println("X1: "+io.getX1());
//                    System.out.println("Y1: "+io.getY1());
//                    System.out.println("X2: "+io.getX2());
//                    System.out.println("Y2: "+io.getY2());
                }
            }

            this.iter = new FileFromPathIterator(this.inputSplit.locationsPathIterator());
            if (split instanceof FileSplit) {
                FileSplit split1 = (FileSplit)split;
                this.labels.remove(split1.getRootDir());
            }

            this.labels = new ArrayList(labelSet);
            Collections.sort(this.labels);
        } else {
            throw new IllegalArgumentException("No path locations found in the split.");
        }
    }

    public List<List<Writable>> next(int num) {
        List<File> files = new ArrayList(num);
        List<List<ImageObject>> objects = new ArrayList(num);

        int nClasses;
        for(nClasses = 0; nClasses < num && this.hasNext(); ++nClasses) {
            File f = (File)this.iter.next();
            this.currentFile = f;
            if (!f.isDirectory()) {
                files.add(f);
                objects.add(this.labelProvider.getImageObjectsForPath(f.getPath()));
            }
        }

        nClasses = this.labels.size();
        INDArray outImg = Nd4j.create(new long[]{(long)files.size(), this.channels, this.height, this.width});
        INDArray outLabel = Nd4j.create(new int[]{files.size(), 4 + nClasses, this.gridH, this.gridW});
        int exampleNum = 0;

        for(int i = 0; i < files.size(); ++i) {
            File imageFile = (File)files.get(i);
            this.currentFile = imageFile;

            try {
                this.invokeListeners(imageFile);
                Image image = this.imageLoader.asImageMatrix(imageFile);
                this.currentImage = image;
                Nd4j.getAffinityManager().ensureLocation(image.getImage(), Location.DEVICE);
                outImg.put(new INDArrayIndex[]{NDArrayIndex.point((long)exampleNum), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, image.getImage());
//                The method declared at bottom
                List<ImageObject> objectsThisImg = (List)objects.get(exampleNum);
                this.label(image, objectsThisImg, outLabel, exampleNum);
//                System.out.println("Image: "+image);
                System.out.println("Imagefile name: "+ imageFile);
                System.out.println("ObjectThisImg: "+objectsThisImg);

                for (ImageObject objectThisImg : objectsThisImg) {
                    System.out.println(objectThisImg.getLabel());
                    System.out.println(objectThisImg.getX1());
                    System.out.println(objectThisImg.getY1());
                    System.out.println(objectThisImg.getX2());
                    System.out.println(objectThisImg.getY2());
                }

            } catch (IOException var12) {
                throw new RuntimeException(var12);
            }

            ++exampleNum;
        }

        return new NDArrayRecordBatch(Arrays.asList(outImg, outLabel));
    }

    private void label(Image image, List<ImageObject> objectsThisImg, INDArray outLabel, int exampleNum) {
        int oW = image.getOrigW();
        int oH = image.getOrigH();
        int W = oW;
        int H = oH;
        Iterator var9 = objectsThisImg.iterator();

        while(true) {
            ImageObject io;
            double cx;
            double cy;
            int minY;
            int maxY;
            do {
                if (!var9.hasNext()) {
                    return;
                }

                io = (ImageObject)var9.next();
                cx = io.getXCenterPixels();
                cy = io.getYCenterPixels();
                if (this.imageTransform == null) {
                    break;
                }

                W = this.imageTransform.getCurrentImage().getWidth();
                H = this.imageTransform.getCurrentImage().getHeight();
                float[] pts = this.imageTransform.query(new float[]{(float)io.getX1(), (float)io.getY1(), (float)io.getX2(), (float)io.getY2()});
                int minX = Math.round(Math.min(pts[0], pts[2]));
                int maxX = Math.round(Math.max(pts[0], pts[2]));
                minY = Math.round(Math.min(pts[1], pts[3]));
                maxY = Math.round(Math.max(pts[1], pts[3]));
                io = new ImageObject(minX, minY, maxX, maxY, io.getLabel());
                cx = io.getXCenterPixels();
                cy = io.getYCenterPixels();
            } while(cx < 0.0D || cx >= (double)W || cy < 0.0D || cy >= (double)H);
            //tl =TopLeft , br = BottomRight
            double[] cxyPostScaling = ImageUtils.translateCoordsScaleImage(cx, cy, (double)W, (double)H, (double)this.width, (double)this.height);
            double[] tlPost = ImageUtils.translateCoordsScaleImage((double)io.getX1(), (double)io.getY1(), (double)W, (double)H, (double)this.width, (double)this.height);
            double[] brPost = ImageUtils.translateCoordsScaleImage((double)io.getX2(), (double)io.getY2(), (double)W, (double)H, (double)this.width, (double)this.height);
            minY = (int)(cxyPostScaling[0] / (double)this.width * (double)this.gridW);
            maxY = (int)(cxyPostScaling[1] / (double)this.height * (double)this.gridH);
            tlPost[0] = tlPost[0] / (double)this.width * (double)this.gridW;
            tlPost[1] = tlPost[1] / (double)this.height * (double)this.gridH;
            brPost[0] = brPost[0] / (double)this.width * (double)this.gridW;
            brPost[1] = brPost[1] / (double)this.height * (double)this.gridH;
            Preconditions.checkState(maxY >= 0 && (long)maxY < outLabel.size(2), "Invalid image center in Y axis: calculated grid location of %s, must be between 0 (inclusive) and %s (exclusive). Object label center is outside of image bounds. Image object: %s", maxY, outLabel.size(2), io);
            Preconditions.checkState(minY >= 0 && (long)minY < outLabel.size(3), "Invalid image center in X axis: calculated grid location of %s, must be between 0 (inclusive) and %s (exclusive). Object label center is outside of image bounds. Image object: %s", maxY, outLabel.size(2), io);
            outLabel.putScalar((long)exampleNum, 0L, (long)maxY, (long)minY, tlPost[0]);
            outLabel.putScalar((long)exampleNum, 1L, (long)maxY, (long)minY, tlPost[1]);
            outLabel.putScalar((long)exampleNum, 2L, (long)maxY, (long)minY, brPost[0]);
            outLabel.putScalar((long)exampleNum, 3L, (long)maxY, (long)minY, brPost[1]);
            int labelIdx = this.labels.indexOf(io.getLabel());
            outLabel.putScalar((long)exampleNum, (long)(4 + labelIdx), (long)maxY, (long)minY, 1.0D);
            System.out.println("Label: "+io.getLabel());
            System.out.println("X1: "+tlPost[0]);
            System.out.println("Y1: "+tlPost[1]);
            System.out.println("X2: "+brPost[0]);
            System.out.println("Y2: "+brPost[1]);
        }
    }

    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        this.invokeListeners(uri);
        if (this.imageLoader == null) {
            this.imageLoader = new NativeImageLoader(this.height, this.width, this.channels, this.imageTransform);
        }

        Image image = this.imageLoader.asImageMatrix(dataInputStream);
        Nd4j.getAffinityManager().ensureLocation(image.getImage(), Location.DEVICE);
        List<Writable> ret = RecordConverter.toRecord(image.getImage());
        if (this.appendLabel) {
            List<ImageObject> imageObjectsForPath = this.labelProvider.getImageObjectsForPath(uri.getPath());
            int nClasses = this.labels.size();
            INDArray outLabel = Nd4j.create(new int[]{1, 4 + nClasses, this.gridH, this.gridW});
            //label being used
            this.label(image, imageObjectsForPath, outLabel, 0);
            ret.add(new NDArrayWritable(outLabel));
        }

        return ret;
    }

    public Record nextRecord() {
        List<Writable> list = this.next();
        URI uri = URIUtil.fileToURI(this.currentFile);
        return new org.datavec.api.records.impl.Record(list, new RecordMetaDataImageURI(uri, BaseImageRecordReader.class, this.currentImage.getOrigC(), this.currentImage.getOrigH(), this.currentImage.getOrigW()));
    }
}
