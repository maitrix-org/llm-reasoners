(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c d h i)
(:init 
(handempty)
(ontable c)
(ontable d)
(ontable h)
(ontable i)
(clear c)
(clear d)
(clear h)
(clear i)
)
(:goal
(and
(on c d)
(on d h)
(on h i)
)))