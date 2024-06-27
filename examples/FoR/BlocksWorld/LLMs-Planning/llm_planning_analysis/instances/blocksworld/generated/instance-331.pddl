(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h j l)
(:init 
(handempty)
(ontable h)
(ontable j)
(ontable l)
(clear h)
(clear j)
(clear l)
)
(:goal
(and
(on h j)
(on j l)
)))