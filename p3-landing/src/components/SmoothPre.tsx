import React from 'react'
import { type AnnotationHandler, InnerToken, Pre } from 'codehike/code'
import {
  type TokenTransitionsSnapshot,
  calculateTransitions,
  getStartingSnapshot,
} from 'codehike/utils/token-transitions'

const MAX_TRANSITION_DURATION = 1600 // ms — slower than docs default (900) so the
// morph is visible during natural scroll. With short duration, each token's
// animation completes between scroll keyframes and the user perceives a cut.

type PreProps = React.ComponentProps<typeof Pre>

const inlineBlockToken: AnnotationHandler = {
  name: 'inline-block',
  Token: (props) => (
    <InnerToken merge={props} style={{ display: 'inline-block' }} />
  ),
}

export class SmoothPre extends React.Component<PreProps> {
  ref: React.RefObject<HTMLPreElement>

  constructor(props: PreProps) {
    super(props)
    this.ref = React.createRef<HTMLPreElement>()
  }

  render() {
    const handlers = [inlineBlockToken, ...(this.props.handlers || [])]
    const style = { ...this.props.style, position: 'relative' as const }
    return <Pre ref={this.ref} {...this.props} style={style} handlers={handlers} />
  }

  getSnapshotBeforeUpdate() {
    return getStartingSnapshot(this.ref.current!)
  }

  componentDidUpdate(
    _prevProps: PreProps,
    _prevState: never,
    snapshot: TokenTransitionsSnapshot,
  ) {
    const transitions = calculateTransitions(this.ref.current!, snapshot)
    transitions.forEach(({ element, keyframes, options }) => {
      const { translateX, translateY, ...kf } = keyframes as any
      if (translateX && translateY) {
        kf.translate = [
          `${translateX[0]}px ${translateY[0]}px`,
          `${translateX[1]}px ${translateY[1]}px`,
        ]
      }
      element.animate(kf, {
        duration: options.duration * MAX_TRANSITION_DURATION,
        delay: options.delay * MAX_TRANSITION_DURATION,
        easing: options.easing,
        fill: 'both',
      })
    })
  }
}
