(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k f j h)
(:init 
(handempty)
(ontable k)
(ontable f)
(ontable j)
(ontable h)
(clear k)
(clear f)
(clear j)
(clear h)
)
(:goal
(and
(on k f)
(on f j)
(on j h)
)))