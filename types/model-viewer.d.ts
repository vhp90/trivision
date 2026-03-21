declare namespace JSX {
  interface IntrinsicElements {
    'model-viewer': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
      src?: string;
      alt?: string;
      autoplay?: boolean;
      ar?: boolean;
      cameraControls?: boolean;
      environmentImage?: string;
      exposure?: string;
      shadowIntensity?: string;
      touchAction?: string;
    };
  }
}
