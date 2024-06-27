(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects d b i)
(:init 
(handempty)
(ontable d)
(ontable b)
(ontable i)
(clear d)
(clear b)
(clear i)
)
(:goal
(and
(on d b)
(on b i)
)))