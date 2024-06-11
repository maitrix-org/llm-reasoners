(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b l f e)
(:init 
(handempty)
(ontable b)
(ontable l)
(ontable f)
(ontable e)
(clear b)
(clear l)
(clear f)
(clear e)
)
(:goal
(and
(on b l)
(on l f)
(on f e)
)))