(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k h l)
(:init 
(handempty)
(ontable k)
(ontable h)
(ontable l)
(clear k)
(clear h)
(clear l)
)
(:goal
(and
(on k h)
(on h l)
)))