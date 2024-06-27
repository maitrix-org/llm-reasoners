(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects f j g)
(:init 
(handempty)
(ontable f)
(ontable j)
(ontable g)
(clear f)
(clear j)
(clear g)
)
(:goal
(and
(on f j)
(on j g)
)))