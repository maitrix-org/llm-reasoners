(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e d b i)
(:init 
(handempty)
(ontable e)
(ontable d)
(ontable b)
(ontable i)
(clear e)
(clear d)
(clear b)
(clear i)
)
(:goal
(and
(on e d)
(on d b)
(on b i)
)))